"""单路故障降级。

任意一路出故障（异常/超时/索引缺失）时，整体检索**仍要返回结果**，只是
该路缺席。不能让一条路的故障拖垮整个查询。

降级只在**三路检索节点内部**捕获——不吞掉关键路径的异常。
"""

import pytest

from ragv1.types import Hit
from ragv1.retrieve import ALL_PATHS, Retriever
from ragv1.orchestration.supervisor import build_graph, run_query, run_query_meta


# 路径名 → Retriever 的构造参数名。
# 注意二者并不相同：全文路的**路径名是 fulltext，属性名是 fts**。
_KWARG = {"vector": "vector", "fulltext": "fts", "graph": "graph"}


class Boom:
    def __init__(self, name):
        self.name = name

    def retrieve(self, query, k):
        raise RuntimeError(f"{self.name} 挂了")


class Ok:
    def __init__(self, name):
        self.name = name

    def retrieve(self, query, k):
        return [Hit(self.name + "-c", 1, 1.0, self.name)]


def mk(bad):
    kwargs = {_KWARG[p]: (Boom(p) if p == bad else Ok(p)) for p in ALL_PATHS}
    return build_graph(Retriever(**kwargs), ALL_PATHS)


@pytest.mark.parametrize("bad", sorted(ALL_PATHS))
def test_any_single_path_failure_still_returns(bad):
    out = run_query(mk(bad), "q", k=5)
    assert out, f"{bad} 故障时整体返回为空"
    assert all(bad not in h.sources for h in out)


@pytest.mark.parametrize("bad", sorted(ALL_PATHS))
def test_failure_is_reported_to_caller(bad):
    """调用方要能看出「哪一路被跳过了」，而不是只拿到一个变小的结果集。"""
    _, degraded = run_query_meta(mk(bad), "q", k=5)
    assert bad in degraded
    assert degraded[bad]


def test_healthy_paths_unaffected():
    out = run_query(mk("graph"), "q", k=5)
    sources = {s for h in out for s in h.sources}
    assert sources == {"vector", "fulltext"}


def test_no_degradation_when_all_healthy():
    all_ok = {_KWARG[p]: Ok(p) for p in ALL_PATHS}
    _, degraded = run_query_meta(build_graph(Retriever(**all_ok), ALL_PATHS), "q", k=5)
    assert degraded == {}


def test_on_error_raise_propagates():
    with pytest.raises(RuntimeError):
        run_query(mk("graph"), "q", k=5, on_error="raise")
