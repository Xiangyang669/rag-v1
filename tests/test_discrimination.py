"""评估集区分度自检。

**评测前**先验证评估集有没有区分力：若某类问题上三路的 recall@5 差值过小，
说明该类测不出差异，需加强或删除。

一个测不出差异的评估集比没有更危险——它会给出「看起来有数」的假结论。
"""

from ragv1.evaluation.harness import Matrix, check_discrimination


def mk(cells):
    return Matrix(cells=cells)


FLAT_A = {
    ("A", "vector"): {"recall@5": 0.90},
    ("A", "fulltext"): {"recall@5": 0.88},
    ("A", "graph"): {"recall@5": 0.89},
    ("A", "fused"): {"recall@5": 0.90},
}

SEPARATED_C = {
    ("C", "vector"): {"recall@5": 0.20},
    ("C", "fulltext"): {"recall@5": 0.22},
    ("C", "graph"): {"recall@5": 0.80},
    ("C", "fused"): {"recall@5": 0.82},
}


def test_flat_category_is_flagged():
    warns = check_discrimination(mk(FLAT_A), threshold=0.05)
    assert len(warns) == 1
    assert "A" in warns[0]


def test_separated_category_is_not_flagged():
    assert check_discrimination(mk(SEPARATED_C), threshold=0.05) == []


def test_message_states_magnitude():
    assert "0.0" in check_discrimination(mk(FLAT_A), threshold=0.05)[0]


def test_only_flat_categories_are_flagged():
    warns = check_discrimination(mk({**FLAT_A, **SEPARATED_C}), threshold=0.05)
    assert len(warns) == 1
    assert "A" in warns[0]


def test_threshold_is_respected():
    # 差值 0.02：阈值 0.01 时不该报，0.05 时报
    assert check_discrimination(mk(FLAT_A), threshold=0.01) == []
    assert len(check_discrimination(mk(FLAT_A), threshold=0.05)) == 1


def test_category_with_too_few_paths_is_skipped():
    """只有一路有数据时无法谈"差异"，不应误报。"""
    m = mk({("B", "vector"): {"recall@5": 0.5}})
    assert check_discrimination(m, threshold=0.05) == []


def test_empty_matrix_yields_no_warnings():
    assert check_discrimination(mk({}), threshold=0.05) == []
