"""Langfuse 埋点。

两条要求：
1. **没配置 key 时绝不报错**——本地开发/CI 不应该因为没有 Langfuse 跑不起来
2. 有 key 时，一次查询能在 Langfuse 上看到三路各自的 span
"""

from contextlib import contextmanager

import pytest

from ragv1 import observability
from ragv1.types import Hit
from ragv1.retrieve import ALL_PATHS, Retriever
from ragv1.orchestration.supervisor import build_graph, run_query

# 路径名 → Retriever 构造参数名（全文路路径名 fulltext、属性名 fts）
_KWARG = {"vector": "vector", "fulltext": "fts", "graph": "graph"}


class Ok:
    def __init__(self, name):
        self.name = name

    def retrieve(self, query, k):
        return [Hit(self.name + "-c", 1, 1.0, self.name)]


def make_retriever():
    return Retriever(**{_KWARG[p]: Ok(p) for p in ALL_PATHS})


@contextmanager
def _recorder(bucket, name):
    bucket.append(name)
    yield


def test_tracer_without_keys_is_noop(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = observability.tracer()
    with observability.span(t, "probe"):
        pass  # 不抛异常即通过


def test_run_query_emits_span_per_path(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(observability, "span", lambda t, n: _recorder(recorded, n))

    run_query(build_graph(make_retriever(), ALL_PATHS), "q", k=5)

    assert {"vector", "fulltext", "graph"} <= set(recorded)


def test_tracing_does_not_change_results(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(observability, "span", lambda t, n: _recorder(recorded, n))
    out = run_query(build_graph(make_retriever(), ALL_PATHS), "q", k=5)
    assert {s for h in out for s in h.sources} == {"vector", "fulltext", "graph"}


# ── 以下两条针对一个真实缺口 ────────────────────────────────
# 无 key 时 tracer() 返回 None，span() 走 no-op 分支 —— 真实客户端这条路径
# 在其余测试里**从未被触达**。本任务第一版正是在这里写错了方法名
# （用了 start_as_current_span，而 langfuse v4 叫
#  start_as_current_observation），测试全绿但生产路径一调用就崩。


class _RealApiClient:
    """只实现 langfuse v4 的**真实** API。若 span() 写错方法名，这里会
    AttributeError —— 这正是我们要它能抓住的。"""

    def __init__(self):
        self.spans: list[str] = []

    @contextmanager
    def start_as_current_observation(self, *, name, as_type="span"):
        self.spans.append(name)
        yield None


def test_span_uses_the_real_langfuse_api_name():
    client = _RealApiClient()
    with observability.span(client, "vector"):
        pass
    assert client.spans == ["vector"]


def test_installed_langfuse_client_exposes_the_api_we_use():
    """版本漂移护栏：装着的 SDK 必须真的提供我们调用的那个方法。"""
    from langfuse import get_client

    client = get_client()
    assert hasattr(client, "start_as_current_observation")
    ctx = client.start_as_current_observation(name="probe")
    assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")
