"""评估集构成要求的离线校验。

这些断言是评估集的"合同"：只读 JSONL + `load_qaset`，不碰索引、
不联网、不读任何 key。无论 `.indexes/kb_zh` 是否存在都必须能跑、且结果确定。

调参只能动 dev，test 是 held-out——因此 dev/test 不允许有重复问题。
"""

from pathlib import Path

import pytest

from ragv1.evaluation.qaset import load_qaset

SETS = Path(__file__).resolve().parents[1] / "ragv1" / "evaluation" / "qa_sets"

ALLOWED_CATEGORIES = {
    "fact",
    "multi_hop",
    "comparison",
    "temporal",
    "unanswerable",
    "multi_turn",
}


@pytest.mark.parametrize("name", ["zh_dev.jsonl", "zh_test.jsonl"])
def test_set_meets_composition_requirements(name):
    items = load_qaset(SETS / name)
    cats = {i.category for i in items}

    assert len(items) >= 15, f"{name} 条数不足（{len(items)}）"
    assert "unanswerable" in cats, f"{name} 缺不可回答题（测拒答必需）"
    assert cats & {"multi_hop", "comparison", "temporal"}, f"{name} 缺多跳/比较/时序题"

    unanswerable = [i for i in items if i.category == "unanswerable"]
    assert len(unanswerable) >= 3, f"{name} 不可回答题不足 3 条（{len(unanswerable)}）"


@pytest.mark.parametrize("name", ["zh_dev.jsonl", "zh_test.jsonl"])
def test_categories_and_answer_ids_consistent(name):
    items = load_qaset(SETS / name)
    for i in items:
        assert i.category in ALLOWED_CATEGORIES, f"{name} 未知分类：{i.category}"
        if i.category == "unanswerable":
            assert i.answer_chunk_ids == (), f"{name} 不可回答题不应有答案块：{i.question}"
        else:
            assert i.answer_chunk_ids, f"{name} 可回答题缺答案块：{i.question}"


def test_dev_and_test_do_not_overlap():
    dev = {i.question for i in load_qaset(SETS / "zh_dev.jsonl")}
    test = {i.question for i in load_qaset(SETS / "zh_test.jsonl")}
    assert not (dev & test), "dev/test 有重复问题，调参会泄漏"


def test_dev_has_multi_turn_with_history():
    items = load_qaset(SETS / "zh_dev.jsonl")
    mt = [i for i in items if i.category == "multi_turn" and i.history]
    assert len(mt) >= 2, f"zh_dev 至少需要 2 条带 history 的多轮题（{len(mt)}）"
