"""V2 验收脚本的检索表部分（任务 8）。离线、确定性。"""

import importlib.util
from pathlib import Path

from ragv1.evaluation.qaset import EvalItem
from ragv1.types import FusedHit


def _load():
    spec = importlib.util.spec_from_file_location(
        "eval_v2", Path(__file__).resolve().parents[1] / "scripts" / "eval_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_retrieval_table_lists_every_path_row():
    mod = _load()
    rows = {
        "vector": [FusedHit("a", 0.1, ("vector",), 1, {})],
        "fused_static": [FusedHit("a", 0.2, ("vector",), 1, {})],
    }
    table = mod.retrieval_table(rows, ["a"])
    assert "| vector |" in table and "| fused_static |" in table


def test_retrieval_table_tolerates_swapped_argument_order():
    # 计划原文（R7）把 T18 的调用写成 retrieval_table(["a"], rows)——容忍它。
    mod = _load()
    rows = {"vector": [FusedHit("a", 0.1, ("vector",), 1, {})]}
    assert mod.retrieval_table(["a"], rows) == mod.retrieval_table(rows, ["a"])


def test_retrieval_table_scores_a_hit():
    mod = _load()
    rows = {"vector": [FusedHit("a", 0.1, ("vector",), 1, {})]}
    table = mod.retrieval_table(rows, ["a"])
    assert "1.000" in table  # recall@5 / recall@10 / mrr 均命中


def test_retrieval_items_excludes_unanswerable():
    mod = _load()
    items = (EvalItem("q1", ("a",), "fact"), EvalItem("q2", (), "unanswerable"))
    assert [it.question for it in mod.retrieval_items(items)] == ["q1"]
