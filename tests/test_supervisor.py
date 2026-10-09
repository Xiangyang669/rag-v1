"""LangGraph 编排层。

**最重要的一条**：三层是**并行通路**，不是串行阶段。Supervisor 同时发起
三路，不是「一路不行再试下一路」。这里用计时来证明——三路各睡 0.2s，
串行需要 0.6s，并行应显著更短。

编排层只负责调度，不含检索逻辑本身。
"""

import time

import pytest

from ragv1.types import Hit
from ragv1.retrieve import ALL_PATHS, Retriever
from ragv1.orchestration.supervisor import build_graph, run_query

SLEEP = 0.2


class Slow:
    """每路睡 SLEEP 秒。串行 3 路 = 0.6s；并行应接近 0.2s。"""

    def __init__(self, name):
        self.name = name
        self.calls = 0

    def retrieve(self, query, k):
        self.calls += 1
        time.sleep(SLEEP)
        return [Hit(self.name + "-c", 1, 1.0, self.name)]


@pytest.fixture
def graph():
    r = Retriever(vector=Slow("vector"), fts=Slow("fulltext"), graph=Slow("graph"))
    return build_graph(r, ALL_PATHS)


def test_three_paths_run_in_parallel(graph):
    t0 = time.perf_counter()
    run_query(graph, "q", k=5)
    elapsed = time.perf_counter() - t0
    assert elapsed < SLEEP * 2.25, f"耗时 {elapsed:.2f}s，看起来是串行（3×{SLEEP}s）"


def test_all_three_paths_contribute(graph):
    sources = {s for h in run_query(graph, "q", k=5) for s in h.sources}
    assert sources == {"vector", "fulltext", "graph"}


def test_only_requested_paths_are_scheduled():
    fakes = {p: Slow(p) for p in ALL_PATHS}
    g = build_graph(
        Retriever(vector=fakes["vector"], fts=fakes["fulltext"], graph=fakes["graph"]),
        frozenset({"graph"}),
    )
    out = run_query(g, "q", k=5)
    assert {s for h in out for s in h.sources} == {"graph"}
    assert fakes["vector"].calls == 0
    assert fakes["fulltext"].calls == 0
    assert fakes["graph"].calls == 1


# ── V2：融合节点接入查询自适应权重（任务 7）──────────────────────────

def test_fuse_node_records_route_weights(monkeypatch):
    """融合节点必须把路由权重写回 state —— 这是调参时唯一的观测点。"""
    from ragv1.orchestration import supervisor
    out = supervisor._fuse_node({
        "query": "MAX_EMBED_BATCH 是多少",
        "ranked": {"fulltext": [Hit(chunk_id="a", rank=1, score=-9.0, path="fulltext")],
                   "vector": [Hit(chunk_id="b", rank=1, score=0.9, path="vector")]},
        "on_error": "skip",
    })
    assert out["weights"]["fulltext"] > 0
    assert len(out["fused"]) == 2


def test_fuse_node_with_empty_ranked_returns_empty():
    from ragv1.orchestration import supervisor
    out = supervisor._fuse_node({"query": "q", "ranked": {}, "on_error": "skip"})
    assert out["fused"] == []
