"""章节层级边（`belongs_to`）——把 heading_path 的层级落成图里真正的边。

在此之前，`_chunk_graph` 只维护 `parent` 用于同名实体的命名限定，**从不
产出任何层级关系**；唯一被构造的关系是 `has_parameter`，只在 `name(params)`
签名标题上产出。中文标题是白话，永远解析不出签名 → `edges` 表 0 行 →
`GraphStore.neighbors()` 永远为空 → README 称为图谱路独家价值的一跳扩张
从未工作过。

契约：
- 方向「子 belongs_to 父」。
- src/dst 都用**标题实体的 canonical**（与 `parent` 同一套取值），因此
  签名标题会挂到它的限定名/全限定名上，而不是字面标题——保证边与
  section 树同构，不挂到与树脱节的字面串上。
- 顶层标题（`parent is None`）不产边、不抛错。
- parameter（非 `_STRUCTURAL_KINDS`）不得成为 belongs_to 的 src/dst。
"""

import pytest

from ragv1.ingest.extract import extract
from ragv1.types import Chunk


def _chunk(cid, path):
    return Chunk(chunk_id=cid, doc_id="d", heading_path=tuple(path), text="正文")


def _belongs(rels):
    return {(r.src, r.dst) for r in rels if r.kind == "belongs_to"}


def test_top_level_heading_emits_no_edge():
    _, rels = extract([_chunk("c1", ["总览"])])
    assert _belongs(rels) == set()


def test_child_points_to_parent():
    _, rels = extract([_chunk("c1", ["知识库", "检索"])])
    assert _belongs(rels) == {("检索", "知识库")}


def test_three_level_chain_emits_two_edges():
    _, rels = extract([_chunk("c1", ["A", "B", "C"])])
    assert _belongs(rels) == {("B", "A"), ("C", "B")}


def test_duplicate_paths_are_deduplicated():
    _, rels = extract([_chunk("c1", ["A", "B"]), _chunk("c2", ["A", "B"])])
    assert _belongs(rels) == {("B", "A")}


def test_output_is_sorted_deterministically():
    _, r1 = extract([_chunk("c1", ["A", "B", "C"])])
    _, r2 = extract([_chunk("c1", ["A", "B", "C"])])
    assert r1 == r2


def test_neighbors_now_returns_siblings_and_parent(tmp_path):
    """一跳扩张第一次真的有东西——这是改动 A 的全部意义。"""
    from ragv1.store.graph_store import GraphStore

    chunks = [_chunk("c1", ["知识库", "检索"]), _chunk("c2", ["知识库", "导入"])]
    g = GraphStore(tmp_path / "g.db")
    g.build(chunks)
    n = set(g.neighbors("知识库"))
    assert {"检索", "导入"} <= n


# ── canonical 口径：签名标题必须用实体 canonical，不能是字面标题 ──────────
# 若父子用了不同命名口径，边会挂到与 section 树脱节的串上（字面标题
# "get(timeout=None)"），而节点其实叫 "Config.get"。这两条钉住口径一致。


def test_signature_child_uses_entity_canonical_not_literal_heading():
    """父是 section 时，签名子标题按「父.裸名」限定；边用限定名。"""
    _, rels = extract([_chunk("c1", ["Config", "get(timeout=None)"])])
    assert _belongs(rels) == {("Config.get", "Config")}
    # 字面标题不得出现在任何边上
    literals = {s for s, _ in _belongs(rels)} | {d for _, d in _belongs(rels)}
    assert "get(timeout=None)" not in literals


def test_grandchild_points_to_qualified_function_canonical():
    """孙标题的父是函数节点，但 dst 必须用它的限定名（与 `parent` 同口径）。"""
    _, rels = extract([_chunk("c1", ["Config", "get(timeout=None)", "Timeout"])])
    assert _belongs(rels) == {("Config.get", "Config"), ("Timeout", "Config.get")}


def test_parameters_do_not_appear_in_belongs_to_edges():
    """parameter 是函数的附属物，不是标题层级；不得成为 belongs_to 的 src/dst。"""
    _, rels = extract([_chunk("c1", ["Config", "get(timeout=None)", "Timeout"])])
    nodes = {s for s, _ in _belongs(rels)} | {d for _, d in _belongs(rels)}
    assert "timeout" not in nodes
    # parameter 只以 has_parameter 的 dst 出现，不被层级边劫持
    assert all(r.kind == "has_parameter" for r in rels if "timeout" in (r.src, r.dst))


def test_identifiers_do_not_appear_in_belongs_to_edges():
    """上个任务用 parameter 代打的那条断言——这里是真正的 identifier 版本。

    identifier 来自正文，不是标题层级；混进 `belongs_to` 会让后续标题拿它
    做命名限定，整条 section 树被污染。断言必须**真的能失败**：先钉住确实
    抽到了标识符、且确实产出了层级边（否则断言在空转）。
    """
    chunks = [
        Chunk(
            "c1",
            "d",
            ("配置", "环境变量"),
            "设 MULTIPLE_DATA_TO_BASE64 为 true",
        )
    ]
    entities, rels = extract(chunks)
    ids = {e.canonical for e in entities if e.kind == "identifier"}
    assert ids == {"MULTIPLE_DATA_TO_BASE64"}, "前提：应当恰好抽出一个标识符"

    belongs = [r for r in rels if r.kind == "belongs_to"]
    endpoints = {r.src for r in belongs} | {r.dst for r in belongs}
    assert endpoints == {"配置", "环境变量"}, "前提：层级边必须存在，否则断言空转"
    assert endpoints & ids == set()
    # 标识符只以 mentioned_in 的 src 出现，不被层级边劫持
    assert all(
        r.kind == "mentioned_in" for r in rels if "MULTIPLE_DATA_TO_BASE64" in (r.src, r.dst)
    )
