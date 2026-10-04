"""端到端验收报告。

交付物的最后一件：一份能**对照 spec 验收判据逐条打勾**的报告，而不是
一堆数字。三条判据来自 spec §10：

1. 融合在 C/D 类上相对最强单路有可测提升，且 A/B 类不掉
2. 评估集区分度自检通过
3. 每条通路的失败模式可复现、可解释
"""

from ragv1.evaluation.harness import Matrix
from ragv1.evaluation.report import build_report

M = Matrix(
    cells={
        ("C", "vector"): {"recall@5": 0.20, "recall@10": 0.30, "mrr": 0.20},
        ("C", "fulltext"): {"recall@5": 0.22, "recall@10": 0.31, "mrr": 0.21},
        ("C", "graph"): {"recall@5": 0.80, "recall@10": 0.85, "mrr": 0.75},
        ("C", "fused"): {"recall@5": 0.82, "recall@10": 0.88, "mrr": 0.80},
        ("A", "vector"): {"recall@5": 0.90, "recall@10": 0.95, "mrr": 0.85},
        ("A", "fulltext"): {"recall@5": 0.91, "recall@10": 0.95, "mrr": 0.86},
        ("A", "graph"): {"recall@5": 0.30, "recall@10": 0.40, "mrr": 0.28},
        ("A", "fused"): {"recall@5": 0.91, "recall@10": 0.96, "mrr": 0.87},
        ("D", "vector"): {"recall@5": 0.10, "recall@10": 0.15, "mrr": 0.10},
        ("D", "fulltext"): {"recall@5": 0.12, "recall@10": 0.16, "mrr": 0.11},
        ("D", "graph"): {"recall@5": 0.70, "recall@10": 0.75, "mrr": 0.65},
        ("D", "fused"): {"recall@5": 0.74, "recall@10": 0.80, "mrr": 0.70},
    }
)


def test_report_contains_matrix_and_gain():
    out = build_report(M, warnings=[])
    assert "recall@5" in out
    assert "fused" in out
    assert "C" in out


def test_report_states_each_acceptance_criterion():
    out = build_report(M, warnings=[])
    assert out.count("判据") >= 3


def test_report_surfaces_warnings():
    out = build_report(M, warnings=["类别 X 无区分力（差 0.01）"])
    assert "无区分力" in out


def test_report_notes_known_bias():
    """报告必须声明评估集的偏差**方向**。

    早期版本写的是「全文路占便宜、图谱增益被低估」——真跑之后实测**方向相反**：
    A/C/D 三类问题由图谱索引的同一套 heading 结构生成，因此偏向图谱路。
    这条测试因此从「含"词面"」改为断言修正后的方向。
    """
    out = build_report(M, warnings=[])
    assert "结构偏向" in out
    assert "偏向图谱路" in out


def test_report_notes_rrf_equal_weight_limitation():
    """等权融合会稀释强路 —— 必须写明，并标为 V2 候选。"""
    out = build_report(M, warnings=[])
    assert "等权" in out
    assert "V2" in out


def test_gain_is_relative_to_strongest_single_path():
    """C 类最强单路是图谱 0.80，融合 0.82 → 增益 +0.020。"""
    out = build_report(M, warnings=[])
    assert "+0.020" in out


def test_category_that_drops_is_flagged():
    dropped = Matrix(
        cells={
            ("A", "vector"): {"recall@5": 0.90, "mrr": 0.9},
            ("A", "fulltext"): {"recall@5": 0.90, "mrr": 0.9},
            ("A", "graph"): {"recall@5": 0.90, "mrr": 0.9},
            ("A", "fused"): {"recall@5": 0.50, "mrr": 0.5},
        }
    )
    out = build_report(dropped, warnings=[])
    assert "回退" in out or "下降" in out or "掉" in out


# ── 修复轮：评审发现 4b ──────────────────────────────────────
def test_criterion_1_is_not_vacuously_true_without_cd_categories():
    """没有 C/D 类数据时不能报「通过」。

    `all(... for c in categories if c in GRAPH_CATEGORIES)` 在没有任何
    C/D 的情况下是 all([]) → True，等于在空数据上打绿灯 —— 而这个报告的
    全部职责就是诚实报告。
    """
    a_only = Matrix(
        cells={
            ("A", "vector"): {"recall@5": 0.9, "mrr": 0.9},
            ("A", "fulltext"): {"recall@5": 0.9, "mrr": 0.9},
            ("A", "graph"): {"recall@5": 0.9, "mrr": 0.9},
            ("A", "fused"): {"recall@5": 0.9, "mrr": 0.9},
        }
    )
    line = next(l for l in build_report(a_only, warnings=[]).splitlines() if "判据 1" in l)
    assert "未通过" in line, f"空 C/D 却报通过: {line}"


def test_empty_matrix_reports_not_passed():
    out = build_report(Matrix(cells={}), warnings=[])
    line = next(l for l in out.splitlines() if "判据 1" in l)
    assert "未通过" in line
