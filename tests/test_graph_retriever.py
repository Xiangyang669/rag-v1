"""图谱路检索器。

最大失效模式（spec 风险清单里的 🔴）：用户问法常常是口语化的
（「这个功能怎么设置」），抽不到任何精确实体——此时**必须返回空**，
而不是抛异常。图谱增益在评估时会因此剧烈波动，这是已知且必须显式处理的。
"""

import pytest

from ragv1.types import Chunk, Hit
from ragv1.store.graph_store import GraphStore
from ragv1.retrieve.graph_retriever import GraphRetriever

CH = [
    Chunk("k1", "d", ("requests 模块",), "模块说明"),
    Chunk(
        "k2",
        "d",
        ("requests 模块", "requests.get(url, timeout=None)"),
        "发起 GET 请求。",
    ),
]


@pytest.fixture
def r(tmp_path):
    g = GraphStore(tmp_path / "g.db")
    g.build(CH)
    return GraphRetriever(g, {c.chunk_id: c for c in CH})


def test_hit_by_entity(r):
    hits = r.retrieve("requests.get 怎么用", k=5)
    assert "k2" in [h.chunk_id for h in hits]
    assert all(h.path == "graph" for h in hits)


def test_colloquial_query_without_entities_returns_empty(r):
    assert r.retrieve("这个功能怎么设置", k=5) == []


def test_empty_query_returns_empty(r):
    assert r.retrieve("", k=5) == []


def test_hit_shape_matches_other_paths(r):
    h = r.retrieve("requests.get", k=1)[0]
    assert isinstance(h, Hit)
    assert h.rank == 1
