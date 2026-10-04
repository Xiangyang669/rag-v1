"""向量路的测试。

测试只验 store 的读写管道（写入 / 排序 / rank 从 1 起 / path 标记 / 持久化），
不验 embedding 模型质量——所以注入确定性假向量，不打外部 API。
真实 embedding 的端到端覆盖留给验收报告（T16）。
"""

import pytest

from conftest import fake_embed
from ragv1.types import Chunk
from ragv1.store.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "vec", embed_fn=fake_embed)


def test_search_returns_hits_ranked_from_one(store):
    store.add(
        [
            Chunk("c1", "d", ("超时设置",), "如何设置请求的超时时间 timeout"),
            Chunk("c2", "d", ("配色",), "如何调整界面配色主题"),
        ]
    )
    hits = store.search("怎么设置超时", k=5)
    assert hits[0].chunk_id == "c1"
    assert hits[0].rank == 1
    assert hits[0].path == "vector"


def test_search_on_empty_query_returns_empty(store):
    store.add([Chunk("c1", "d", ("A",), "内容")])
    assert store.search("", k=5) == []


def test_persistence_across_reopen(tmp_path):
    s1 = VectorStore(tmp_path / "vec", embed_fn=fake_embed)
    s1.add([Chunk("c1", "d", ("A",), "内容")])
    s2 = VectorStore(tmp_path / "vec", embed_fn=fake_embed)
    assert [h.chunk_id for h in s2.search("内容", k=5)] == ["c1"]


def test_add_batches_embeddings(tmp_path):
    """入库应该把整批块**一次性**发给 embedding 接口，而不是每块一次请求。

    1978 个块逐条调用 = 1978 次网络往返；批量后约 60 次。这是入库慢的真正原因。
    """
    batches: list[list[str]] = []

    def counting_embed(texts):
        assert isinstance(texts, list), f"应传文本列表，实际是 {type(texts).__name__}"
        batches.append(list(texts))
        return [[float(i + 1), 0.0, 0.0, 1.0] for i in range(len(texts))]

    store = VectorStore(tmp_path / "vec", embed_fn=counting_embed)
    store.add([Chunk(f"c{i}", "d", ("A",), f"正文{i}") for i in range(10)])

    assert len(batches) == 1, f"应 1 次批量调用，实际 {len(batches)} 次"
    assert len(batches[0]) == 10
