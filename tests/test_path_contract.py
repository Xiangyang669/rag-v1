"""三路召回的**格式契约**。

三路必须产出完全同形的 `Hit`，否则 RRF 融合拿到的结构不一致——而融合层
只处理一种形状。这是硬约束一的落地检验。
"""

from conftest import fake_embed
from ragv1.retrieve.fts_retriever import FtsRetriever
from ragv1.retrieve.graph_retriever import GraphRetriever
from ragv1.retrieve.vector_retriever import VectorRetriever
from ragv1.store.fts_store import FtsStore
from ragv1.store.graph_store import GraphStore
from ragv1.store.vector_store import VectorStore
from ragv1.types import Chunk, Hit

QUERY = "Admin Service get 怎么用"
CHUNKS = [
    Chunk("c1", "d", ("Admin Service",), "Admin Service overview of the system"),
    Chunk("c2", "d", ("Admin Service", "get(url)"), "Admin Service get request handling"),
]


def build_three(tmp_path):
    graph = GraphStore(tmp_path / "g.db")
    graph.build(CHUNKS)
    return [
        VectorRetriever(VectorStore(tmp_path / "vec", embed_fn=fake_embed)),
        FtsRetriever(FtsStore(tmp_path / "kb.db")),
        GraphRetriever(graph, {c.chunk_id: c for c in CHUNKS}),
    ]


def test_graph_path_returns_results_on_real_shape(tmp_path):
    """修复缺口②之后，图谱路在这个 query 上必须真的召回——而不是永远空。"""
    _, _, graph_r = build_three(tmp_path)
    assert graph_r.retrieve(QUERY, k=5), "图谱路返回空"


def test_all_three_emit_the_same_hit_type(tmp_path):
    vec_r, fts_r, graph_r = build_three(tmp_path)
    vec_r._store.add(CHUNKS)
    fts_r._store.add(CHUNKS)

    for name, hits in (
        ("vector", vec_r.retrieve(QUERY, k=5)),
        ("fulltext", fts_r.retrieve(QUERY, k=5)),
        ("graph", graph_r.retrieve(QUERY, k=5)),
    ):
        assert hits, f"{name} 路没召回"
        for h in hits:
            assert isinstance(h, Hit), f"{name} 路返回的不是 Hit"
            assert isinstance(h.chunk_id, str) and h.chunk_id
            assert isinstance(h.rank, int) and h.rank >= 1
            assert isinstance(h.score, float)
            assert h.path == name


def test_ranks_are_contiguous_from_one(tmp_path):
    """三路的 rank 都必须从 1 连续递增——RRF 直接拿它算 1/(k+rank)。"""
    vec_r, fts_r, graph_r = build_three(tmp_path)
    vec_r._store.add(CHUNKS)
    fts_r._store.add(CHUNKS)

    for name, hits in (
        ("vector", vec_r.retrieve(QUERY, k=5)),
        ("fulltext", fts_r.retrieve(QUERY, k=5)),
        ("graph", graph_r.retrieve(QUERY, k=5)),
    ):
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1)), name


def test_all_three_return_empty_on_blank_query(tmp_path):
    vec_r, fts_r, graph_r = build_three(tmp_path)
    vec_r._store.add(CHUNKS)
    fts_r._store.add(CHUNKS)

    for name, hits in (
        ("vector", vec_r.retrieve("", k=5)),
        ("fulltext", fts_r.retrieve("", k=5)),
        ("graph", graph_r.retrieve("", k=5)),
    ):
        assert hits == [], f"{name} 路空查询没返回空列表"
