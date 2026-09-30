"""分层指标 harness。

**只输出分层矩阵（四类 × 四路），不提供任何"总 recall"聚合**——
三路会互相抵消，一个平均数字什么都说明不了。

真正的结论形态是：
「融合后在 C/D 类上把 recall@5 从 0.2 拉到 0.75，而 A/B 类不掉。」
"""

import pytest

from ragv1.types import Hit
from ragv1.retrieve import Retriever
from ragv1.evaluation.qa_gen import QAItem
from ragv1.evaluation.harness import EvalConfig, evaluate, render

ROWS = {"vector", "fulltext", "graph", "fused"}

QAS = [QAItem("q1", ("c1",), "A"), QAItem("q2", ("c2",), "C")]


class Stub:
    def __init__(self, hits_by_query):
        self.m = hits_by_query

    def retrieve(self, query, k):
        return self.m.get(query, [])


@pytest.fixture
def retriever():
    return Retriever(
        vector=Stub({"q1": [Hit("c1", 1, 1.0, "vector")], "q2": []}),
        fts=Stub({}),
        graph=Stub({"q2": [Hit("c2", 1, 1.0, "graph")]}),
    )


def test_matrix_has_four_rows_per_category(retriever):
    m = evaluate(retriever, QAS, EvalConfig(k_values=(5, 10)))
    for category in ("A", "C"):
        assert {r for (c, r) in m.cells if c == category} == ROWS


def test_fused_recall_covers_both_paths(retriever):
    m = evaluate(retriever, QAS, EvalConfig(k_values=(5, 10)))
    assert m.cells[("A", "fused")]["recall@5"] == 1.0
    assert m.cells[("C", "fused")]["recall@5"] == 1.0
    assert m.cells[("C", "vector")]["recall@5"] == 0.0


def test_single_path_row_reflects_that_path_only(retriever):
    m = evaluate(retriever, QAS, EvalConfig(k_values=(5,)))
    assert m.cells[("C", "graph")]["recall@5"] == 1.0
    assert m.cells[("A", "graph")]["recall@5"] == 0.0


def test_mrr_is_reciprocal_of_first_hit_rank():
    qas = [QAItem("q", ("c9",), "A")]

    class R:
        def retrieve(self, query, k, paths=None):
            from ragv1.retrieve import ALL_PATHS

            enabled = ALL_PATHS if paths is None else set(paths)
            return {
                p: ([Hit("x", 1, 1.0, p), Hit("c9", 2, 1.0, p)] if p == "vector" else [])
                for p in enabled
            }

    m = evaluate(R(), qas, EvalConfig(k_values=(5,)))
    assert m.cells[("A", "vector")]["mrr"] == pytest.approx(0.5)


def test_evaluation_is_reproducible(retriever):
    a = evaluate(retriever, QAS, EvalConfig(k_values=(5,)))
    b = evaluate(retriever, QAS, EvalConfig(k_values=(5,)))
    assert a.cells == b.cells


def test_render_contains_every_cell(retriever):
    out = render(evaluate(retriever, QAS, EvalConfig(k_values=(5,))))
    assert "vector" in out
    assert "fused" in out
    assert "recall@5" in out


def test_no_overall_aggregate_is_exposed(retriever):
    """刻意不提供"总 recall"——那会掩盖三路差异。"""
    m = evaluate(retriever, QAS, EvalConfig(k_values=(5,)))
    assert not hasattr(m, "overall")
    assert all(isinstance(key, tuple) and len(key) == 2 for key in m.cells)
