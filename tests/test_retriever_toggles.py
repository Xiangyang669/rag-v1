"""三路独立开关。

「关掉图谱后指标掉多少」这类消融结论全靠它——没有开关，就没法证明
任何一路有用。所以除了返回值正确，还要钉住**被禁用的路径根本没被调用**。
"""

import pytest

from ragv1.types import Hit
from ragv1.retrieve import ALL_PATHS, Retriever


class Fake:
    """假检索器：记录被调用次数，避免依赖真实索引。"""

    def __init__(self, name, out):
        self.name = name
        self.out = out
        self.calls = 0

    def retrieve(self, query, k):
        self.calls += 1
        return self.out


def make_fakes():
    return {
        "vector": Fake("vector", [Hit("c1", 1, 0.9, "vector")]),
        "fts": Fake("fulltext", [Hit("c1", 1, -1.0, "fulltext")]),
        "graph": Fake("graph", [Hit("c2", 1, 1.0, "graph")]),
    }


@pytest.fixture
def retriever():
    return Retriever(**make_fakes())


def test_all_paths_enabled(retriever):
    out = retriever.retrieve("q", k=5, paths=ALL_PATHS)
    assert set(out.keys()) == {"vector", "fulltext", "graph"}


def test_single_path(retriever):
    out = retriever.retrieve("q", k=5, paths={"graph"})
    assert set(out.keys()) == {"graph"}
    assert out["graph"][0].chunk_id == "c2"


def test_no_paths_returns_empty_dict(retriever):
    assert retriever.retrieve("q", k=5, paths=set()) == {}


def test_disabled_paths_are_not_called():
    fakes = make_fakes()
    r = Retriever(**fakes)
    r.retrieve("q", k=5, paths={"graph"})
    assert fakes["vector"].calls == 0
    assert fakes["fts"].calls == 0
    assert fakes["graph"].calls == 1


def test_retriever_exposes_sub_retrievers():
    """T9 的 Supervisor 要按路径**独立**调用三个检索器才能真并行；
    若只暴露 retrieve()，扇出会退化成顺序循环。"""
    fakes = make_fakes()
    r = Retriever(**fakes)
    assert r.vector is fakes["vector"]
    assert r.fts is fakes["fts"]
    assert r.graph is fakes["graph"]
