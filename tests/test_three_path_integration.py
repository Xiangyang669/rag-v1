"""三路并行完整链路集成验证。

前面各模块的测试各自是绿的，但仍可能是「跑不起来」的——build_corpus
只建两路那个缺口就是这么漏过去的。所以这里**真跑一遍**：

    小语料 → build_corpus → 三个真检索器 → Retriever → LangGraph → RRF → 结果

不用真实索引（`.indexes/` 是 gitignored、不可复现），改用 tmp_path 里的
小语料 + 假向量，保证可重复。
"""

import pytest

from conftest import fake_embed
from ragv1.ingest.build import FTS_DB, GRAPH_DB, VECTOR_DIR, build_corpus
from ragv1.orchestration.supervisor import build_graph, run_query, run_query_meta
from ragv1.retrieve import ALL_PATHS, Retriever
from ragv1.retrieve.fts_retriever import FtsRetriever
from ragv1.retrieve.graph_retriever import GraphRetriever
from ragv1.retrieve.vector_retriever import VectorRetriever
from ragv1.store.fts_store import FtsStore
from ragv1.store.graph_store import GraphStore
from ragv1.store.vector_store import VectorStore

DOCS = {
    "admin.md": """# Admin Service

The Admin Service manages users and system status.

## get(url, timeout)

Fetch admin status over HTTP.

## update(payload)

Update admin configuration.
""",
    "deploy.md": """# Deployment

How to deploy the service.

## Docker

Container deployment steps.
""",
}

QUERY = "Admin Service get"


@pytest.fixture
def corpus(tmp_path):
    src = tmp_path / "corpus"
    src.mkdir()
    for name, text in DOCS.items():
        (src / name).write_text(text, encoding="utf-8")
    return src


@pytest.fixture
def indexed(corpus, tmp_path):
    idx = tmp_path / "idx"
    build_corpus(corpus, idx, embed_fn=fake_embed)
    return idx


@pytest.fixture
def retriever(indexed, corpus):
    chunks = []
    from ragv1.config import MAX_CHARS
    from ragv1.ingest.chunker import chunk_document
    from ragv1.ingest.parser import parse_document

    for md in sorted(corpus.rglob("*.md")):
        doc = parse_document(
            md.read_text(encoding="utf-8"), doc_id=md.relative_to(corpus).as_posix()
        )
        chunks.extend(chunk_document(doc, MAX_CHARS))

    graph = GraphStore(indexed / GRAPH_DB)
    return Retriever(
        vector=VectorRetriever(VectorStore(indexed / VECTOR_DIR, embed_fn=fake_embed)),
        fts=FtsRetriever(FtsStore(indexed / FTS_DB)),
        graph=GraphRetriever(graph, {c.chunk_id: c for c in chunks}),
    )


def test_three_paths_all_contribute(retriever):
    """三路都必须真的召回——图谱路不许再是空转的。"""
    ranked = retriever.retrieve(QUERY, k=5, paths=ALL_PATHS)
    assert set(ranked.keys()) == {"vector", "fulltext", "graph"}
    for path, hits in ranked.items():
        assert hits, f"{path} 路没召回"


def test_graph_path_not_empty_in_full_flow(retriever):
    """缺口②的端到端检验：修复前图谱路在这个 query 上必然空。"""
    assert retriever.retrieve(QUERY, k=5, paths={"graph"})["graph"]


def test_supervisor_runs_all_three_and_fuses(retriever):
    graph = build_graph(retriever, ALL_PATHS)
    fused = run_query(graph, QUERY, k=5)

    assert fused, "融合结果为空"
    sources = {s for h in fused for s in h.sources}
    assert sources == {"vector", "fulltext", "graph"}, f"实际来源 {sources}"
    assert [h.rank for h in fused] == list(range(1, len(fused) + 1))


def test_no_degradation_in_full_flow(retriever):
    """完整链路不应有任何一路降级。"""
    _, degraded = run_query_meta(build_graph(retriever, ALL_PATHS), QUERY, k=5)
    assert degraded == {}, f"有路径降级：{degraded}"
