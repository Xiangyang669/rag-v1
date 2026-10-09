"""GraphStore 从磁盘加载图：修「一跳扩张在生产路径上恒为空」的缺陷。

背景（Task 23 量出）：磁盘上中文图 `edges` 表已有 4587 行，但检索时一跳扩张
仍不可用——`GraphStore.__init__` 只建一个空 `_g`，把边读进内存的唯一入口是
`build()`，而生产/评估装配路径 `api/server.py::_open` 只构造 `GraphStore`，
从不调 `build()`。于是 `neighbors()` 对全部 section 返回 `[]`。

本任务：`GraphStore(db_path)` 在**构造时**就把 `nodes` / `edges` 载入内存 `_g`，
使 `neighbors()` / `params_of()` 无需先在同一进程里 `build()` 即可工作。

⚠️ 本任务的核心断言是「**不调 build()** 一跳也必须可用」——用**实例 A 写、
实例 B 只读打开**来模拟生产路径，杜绝测试里手搓 `build()` 把缺陷掩盖过去。

用 `tmp_path`（而非 `tempfile.TemporaryDirectory`）：Windows 上 sqlite 连接
持有文件句柄，`TemporaryDirectory` 退出时的 rmtree 会撞 PermissionError——
那是清理竞态，不是本任务的行为断言。仓库既有的图测试也都用 `tmp_path`。
"""

import pytest

from ragv1.store.graph_store import GraphStore
from ragv1.types import Chunk


def _chunk(cid, path):
    return Chunk(chunk_id=cid, doc_id="d", heading_path=tuple(path), text="正文")


def _built_db(path):
    """用实例 A 写入，返回 db 路径——供实例 B 只读打开。

    模拟生产路径：writer 落盘后退出，reader 只构造 GraphStore，不 build。
    """
    a = GraphStore(path)
    a.build([_chunk("c1", ["知识库", "检索"]), _chunk("c2", ["知识库", "导入"])])
    return path


def test_reopened_store_has_neighbors_without_build(tmp_path):
    """⚠️ 本任务的核心：不调 build()，一跳也必须可用。"""
    b = GraphStore(_built_db(tmp_path / "g.db"))
    assert set(b.neighbors("知识库")) >= {"检索", "导入"}


def test_reopened_store_node_kinds_are_preserved(tmp_path):
    """载入必须与 build() 同一套节点构造规则（同一套属性，如 kind）。

    否则「新建的实例」与「build 过的实例」行为分叉——`params_of()` 读
    `nodes[n]['kind'] == 'parameter'`，正是会被这种分叉击中的地方。
    """
    g = GraphStore(_built_db(tmp_path / "g.db"))
    assert g._g.nodes["知识库"].get("kind") == "section"


def test_reopened_store_params_of_works(tmp_path):
    p = _built_db(tmp_path / "g.db")
    GraphStore(p).build([])  # 清空后重开，确认不是靠残留状态
    c = GraphStore(p)
    assert c.params_of("不存在的实体") == []  # 不抛错


def test_empty_db_gives_empty_neighbors_and_does_not_raise(tmp_path):
    s = GraphStore(tmp_path / "empty.db")
    assert s.neighbors("任意") == []
    assert s.params_of("任意") == []


def test_build_still_resets_and_repopulates(tmp_path):
    s = GraphStore(tmp_path / "g.db")
    s.build([_chunk("c1", ["A", "B"])])
    assert s.neighbors("A") == ["B"]
    s.build([_chunk("c2", ["X", "Y"])])
    assert s.neighbors("A") == []  # 旧边必须被清掉
    assert s.neighbors("X") == ["Y"]


def test_build_empty_list_leaves_graph_empty(tmp_path):
    """`build([])` 之后 `_g` 必须为空——不能留下载入时的旧状态。

    先载入一个非空图（构造即加载），再 `build([])`，内存图必须跟着清空。
    """
    p = tmp_path / "g.db"
    GraphStore(p).build([_chunk("c1", ["A", "B"])])
    s = GraphStore(p)  # 构造即加载，此时 _g 非空
    assert s.neighbors("A") == ["B"]
    s.build([])  # 清空
    assert s.neighbors("A") == []
    assert s._g.number_of_nodes() == 0
    assert s._g.number_of_edges() == 0


def test_load_matches_build_construction_rule(tmp_path):
    """两条路径（磁盘载入 vs 同一进程 build）必须给出同一张图。"""
    p = tmp_path / "g.db"
    built = GraphStore(p)
    built.build([_chunk("c1", ["知识库", "检索"]), _chunk("c2", ["知识库", "导入"])])
    loaded = GraphStore(p)

    assert sorted(built._g.nodes()) == sorted(loaded._g.nodes())
    assert sorted(built._g.edges()) == sorted(loaded._g.edges())
    for node in built._g.nodes():
        assert built._g.nodes[node].get("kind") == loaded._g.nodes[node].get("kind")
    for edge in built._g.edges():
        assert built._g.edges[edge].get("kind") == loaded._g.edges[edge].get("kind")


def test_neighbors_does_not_query_db(tmp_path):
    """`neighbors()` 不许查库——那会把一次遍历变成 N 次 SQL。

    载入后应纯内存遍历：这里把连接换成一调用 execute 就炸的替身，neighbors 仍须可用。
    （`sqlite3.Connection.execute` 是只读的 C 方法，patch 不了，故换整个连接对象。）
    """
    g = GraphStore(_built_db(tmp_path / "g.db"))

    class _NoQuery:
        def execute(self, *args, **kwargs):
            raise AssertionError("neighbors() 不得查库")

    g._con = _NoQuery()
    assert set(g.neighbors("知识库")) >= {"检索", "导入"}


def test_production_assembly_path_yields_working_neighbors():
    """端到端护栏：走真实装配路径（不是测试里手搓 build）。

    断言「确实载入了边」而非仅「不抛」——`neighbors()` 恒返回列表，永不抛，
    只断言 `is not None` 无法失败，起不到护栏作用。
    """
    from pathlib import Path

    from ragv1.api.server import _open

    idx = Path(".indexes/kb_zh")
    if not (idx / "graph.db").is_file():
        pytest.skip("中文索引不存在，先跑 scripts/build_index.py")

    retriever, _fts = _open(idx)
    store = retriever.graph._store
    # 真实索引下至少一个实体的一跳非空 —— 证明边被读进了内存图
    assert any(store.neighbors(e) for e in store.aliases()), (
        "生产装配路径下所有实体的一跳都为空 —— 图没被载入内存"
    )
