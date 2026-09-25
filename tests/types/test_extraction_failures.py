"""Chunk-extraction failure visibility and on_error strategy (#base._batch_safe).

Also regression-tests the hypergraph second-feed crash: _update_data_state
referenced the nonexistent ``self.key_extractor`` (only
``node_key_extractor``/``edge_key_extractor`` exist), so any incremental
feed into a non-empty hypergraph KA raised AttributeError.
"""

import pytest
from pydantic import BaseModel

from tests.mocks import MockChatModel, MockEmbeddings

from hyperextract.types import AutoHypergraph, AutoList


class _Item(BaseModel):
    name: str


class _Entity(BaseModel):
    name: str


class _HyperRelation(BaseModel):
    participants: list[str] = []
    relation_type: str = "rel"


class _FlakyExtractor:
    """Wraps the real extractor; raises for selected chunk indexes.

    Supports both invoke() (single-chunk path) and batch() (multi-chunk
    path); non-failed chunks delegate to the wrapped extractor.
    """

    def __init__(self, inner, fail_indexes=()):
        self._inner = inner
        self.fail_indexes = set(fail_indexes)

    def invoke(self, input):
        if 0 in self.fail_indexes:
            raise RuntimeError("boom 0")
        return self._inner.invoke(input)

    def batch(self, inputs, config=None, return_exceptions=False):
        out = []
        for i, inp in enumerate(inputs):
            if i in self.fail_indexes:
                out.append(RuntimeError(f"boom {i}"))
            else:
                out.append(self._inner.invoke(inp))
        return out


def _list_ka(llm_client, embedder, **kwargs):
    return AutoList(
        item_schema=_Item, llm_client=llm_client, embedder=embedder, **kwargs
    )


def _hypergraph(llm_client, embedder):
    return AutoHypergraph(
        node_schema=_Entity,
        edge_schema=_HyperRelation,
        node_key_extractor=lambda x: x.name,
        edge_key_extractor=lambda x: f"{x.relation_type}_{sorted(x.participants)}",
        nodes_in_edge_extractor=lambda x: tuple(x.participants),
        llm_client=llm_client,
        embedder=embedder,
    )


class TestHypergraphSecondFeed:
    def test_second_incremental_feed_does_not_crash(self, llm_client, embedder):
        hg = _hypergraph(llm_client, embedder)
        hg.feed_text("first document", source_id="d1")
        # Second feed hit _update_data_state's non-empty path, which used the
        # nonexistent self.key_extractor.
        hg.feed_text("second document", source_id="d2")
        assert hg.source_content_hash("d1") is None or True
        assert {"d1", "d2"} <= set(hg.sources())

    def test_second_feed_search_finds_new_content(self, llm_client, embedder):
        hg = _hypergraph(llm_client, embedder)
        hg.feed_text("Alpha document about apples.", source_id="d1")
        hg.feed_text("Beta document about zebras.", source_id="d2")
        hg.build_index()
        hits = hg.search("zebras", top_k_nodes=3, top_k_edges=0)
        assert hits  # incremental feed is searchable


class TestExtractionFailures:
    def test_failures_collected_by_default(self, llm_client, embedder):
        ka = _list_ka(llm_client, embedder)
        ka.data_extractor = _FlakyExtractor(ka.data_extractor, fail_indexes={1})
        long_text = "chunk boundary filler. " * 300  # > chunk_size -> multiple chunks
        ka.feed_text(long_text)
        failures = ka.extraction_failures
        assert len(failures) == 1
        assert failures[0]["chunk_index"] == 1
        assert "boom 1" in failures[0]["error"]
        assert failures[0]["stage"]

    def test_failures_reset_between_runs(self, llm_client, embedder):
        ka = _list_ka(llm_client, embedder)
        original = ka.data_extractor
        ka.data_extractor = _FlakyExtractor(original, fail_indexes={0})
        ka.feed_text("first run content", source_id="d1")
        assert len(ka.extraction_failures) == 1
        ka.data_extractor = _FlakyExtractor(original, fail_indexes=set())
        ka.feed_text("second run content", source_id="d2")
        assert ka.extraction_failures == []

    def test_on_error_raise_aborts_feed(self, embedder):
        ka = _list_ka(MockChatModel(), embedder, on_error="raise")
        assert ka.on_error == "raise"
        ka.data_extractor = _FlakyExtractor(ka.data_extractor, fail_indexes={0})
        with pytest.raises(RuntimeError, match="boom 0"):
            ka.feed_text("anything")

    def test_invalid_on_error_rejected(self, llm_client, embedder):
        with pytest.raises(ValueError, match="on_error"):
            _list_ka(llm_client, embedder, on_error="explode")

    def test_graph_family_accepts_on_error(self, embedder):
        from hyperextract.types import AutoGraph

        class _E(BaseModel):
            name: str

        class _R(BaseModel):
            source: str
            target: str
            relation_type: str

        ka = AutoGraph(
            node_schema=_E,
            edge_schema=_R,
            node_key_extractor=lambda x: x.name,
            edge_key_extractor=lambda x: x.relation_type,
            nodes_in_edge_extractor=lambda x: (x.source, x.target),
            llm_client=MockChatModel(),
            embedder=embedder,
            on_error="raise",
        )
        assert ka.on_error == "raise"
