"""评估集自动生成（A/B/C/D 四类）。

设计目标不是"覆盖文档"，而是**让每一路都有自己独占得分的问题**：
- A 词面型 → 考全文
- B 语义型 → 考向量（规则同义替换改写，**不调 LLM**）
- C 聚合型 / D 归属型 → 考图谱（答案是一组块，不是单个块）

答案块一律**由结构推导**，零人工标注。
"""

import pytest

from ragv1.types import Chunk
from ragv1.store.graph_store import GraphStore
from ragv1.evaluation.qa_gen import generate_qas


def mk_chunks():
    return [
        Chunk(f"k{i}", "d", ("requests 模块", f"{name}(url, timeout=None)"), f"{name} 的说明")
        for i, name in enumerate(["requests.get", "requests.post", "requests.put"])
    ]


@pytest.fixture
def gen(tmp_path):
    chunks = mk_chunks()
    graph = GraphStore(tmp_path / "g.db")
    graph.build(chunks)
    return generate_qas(chunks, graph, per_category=5)


def test_all_four_categories_present(gen):
    assert {q.category for q in gen} == {"A", "B", "C", "D"}


def test_answer_chunks_come_from_structure(gen):
    valid = {c.chunk_id for c in mk_chunks()}
    assert all(set(q.answer_chunk_ids) <= valid for q in gen)


def test_category_c_answers_span_multiple_chunks(gen):
    """C 是图谱独占场景：答案必须是**一组**块，不是单个块。"""
    c = [q for q in gen if q.category == "C"]
    assert c
    assert all(len(q.answer_chunk_ids) >= 2 for q in c)


def test_category_d_collects_a_whole_subtree(gen):
    d = [q for q in gen if q.category == "D"]
    assert d
    assert all(len(q.answer_chunk_ids) >= 2 for q in d)


def test_generation_is_deterministic(tmp_path):
    chunks = mk_chunks()
    graph = GraphStore(tmp_path / "g.db")
    graph.build(chunks)
    a = [(q.question, q.answer_chunk_ids, q.category) for q in generate_qas(chunks, graph, 5)]
    b = [(q.question, q.answer_chunk_ids, q.category) for q in generate_qas(chunks, graph, 5)]
    assert a == b


def test_b_category_uses_rule_synonyms_not_llm(gen):
    """B 类必须能在同义词表里找到替换痕迹 —— 证明是规则改写而非模型改写。"""
    from ragv1.evaluation.synonyms import SYNONYMS

    values = {v for vs in SYNONYMS.values() for v in vs}
    b = [q for q in gen if q.category == "B"]
    assert b
    assert all(any(v in q.question for v in values) for q in b)


def test_per_category_limit_is_respected(tmp_path):
    chunks = mk_chunks()
    graph = GraphStore(tmp_path / "g.db")
    graph.build(chunks)
    out = generate_qas(chunks, graph, per_category=1)
    for category in "ABCD":
        assert len([q for q in out if q.category == category]) <= 1


def test_empty_chunks_yield_no_questions(tmp_path):
    graph = GraphStore(tmp_path / "g.db")
    graph.build([])
    assert generate_qas([], graph, per_category=5) == []
