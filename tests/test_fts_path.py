"""全文路的测试。

关键不变量：全文索引与向量索引必须建在**同一套块**上（块 ID 一一对应），
否则后面 RRF 融合时对不上——这是计划里的硬约束一。
"""

import pytest

from conftest import fake_embed
from ragv1.types import Chunk
from ragv1.store.fts_store import FtsStore
from ragv1.store.vector_store import VectorStore
from ragv1.ingest.build import build_corpus

CHUNKS = [
    Chunk("c1", "d", ("超时设置",), "如何设置请求的超时时间 timeout"),
    Chunk("c2", "d", ("配色",), "如何调整界面配色主题"),
]


@pytest.fixture
def fts(tmp_path):
    s = FtsStore(tmp_path / "kb.db")
    s.add(CHUNKS)
    return s


def test_keyword_hit_ranked(fts):
    hits = fts.search("timeout", k=5)
    assert hits[0].chunk_id == "c1"
    assert hits[0].path == "fulltext"


def test_bm25_score_is_negative(fts):
    assert fts.search("timeout", k=5)[0].score < 0


def test_same_chunk_ids_as_vector_path(tmp_path):
    v = VectorStore(tmp_path / "vec", embed_fn=fake_embed)
    v.add(CHUNKS)
    f = FtsStore(tmp_path / "kb.db")
    f.add(CHUNKS)
    assert f.chunk_ids() == {c.chunk_id for c in CHUNKS}


def test_empty_query_returns_empty(fts):
    assert fts.search("", k=5) == []


def test_build_corpus_keeps_two_paths_in_sync(tmp_path):
    """入库时两路必须收到同一个 all_chunks——块 ID 集合完全一致。"""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("# A\n\nx\n", encoding="utf-8")
    (corpus / "b.md").write_text("# B\n\ny\n", encoding="utf-8")

    n = build_corpus(corpus, tmp_path / "idx", embed_fn=fake_embed)

    v = VectorStore(tmp_path / "idx" / "vec", embed_fn=fake_embed)
    f = FtsStore(tmp_path / "idx" / "kb.db")
    assert n == 2
    assert f.chunk_ids() == {h.chunk_id for h in v.search("x y", k=10)} | {
        h.chunk_id for h in v.search("y", k=10)
    }


# ── 修复轮：评审发现 1 ────────────────────────────────────────
def test_reingest_does_not_duplicate_hits(tmp_path):
    """重跑入库不能让同一个块出现两次。

    chunks_fts 用裸 INSERT 且无唯一约束时，第二次 add 会让全文路返回重复
    块 —— 它在 RRF 里拿到 2/(k+rank)，直接污染评估数字。
    """
    s = FtsStore(tmp_path / "kb.db")
    s.add(CHUNKS)
    s.add(CHUNKS)  # 重跑入库（"加了几篇文档再重建"是很自然的操作）

    hits = s.search("timeout", k=10)
    assert len({h.chunk_id for h in hits}) == len(hits), f"出现重复块: {hits}"
    assert len(s.chunk_ids()) == len(CHUNKS)
