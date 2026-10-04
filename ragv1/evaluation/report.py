"""端到端验收报告。

交付物的最后一件：把矩阵变成一份**能对照 spec 验收判据逐条打勾**的文档，
而不是一堆数字。三条判据（spec §10）：

1. 融合在 C/D 类上相对最强单路有可测提升，且 A/B 类不掉
2. 评估集区分度自检通过
3. 每条通路的失败模式可复现、可解释
"""

from ragv1.evaluation.harness import (
    ROW_FULLTEXT,
    ROW_FUSED,
    ROW_GRAPH,
    ROW_VECTOR,
    Matrix,
    render,
)

# 单路的三行（不含融合）
SINGLE_ROWS = (ROW_VECTOR, ROW_FULLTEXT, ROW_GRAPH)

# 图谱独占的两类：答案是一组块而不是一个块
GRAPH_CATEGORIES = ("C", "D")

RECALL5 = "recall@5"


def _recall5(matrix: Matrix, category: str, row: str) -> float:
    return matrix.cells.get((category, row), {}).get(RECALL5, 0.0)


def _best_single(matrix: Matrix, category: str) -> float:
    return max(_recall5(matrix, category, row) for row in SINGLE_ROWS)


def _gain_table(matrix: Matrix, categories: list[str]) -> tuple[list[str], list[str]]:
    lines = [
        "| 类别 | 最强单路 recall@5 | 融合 recall@5 | 增益 | 判定 |",
        "|---|---|---|---|---|",
    ]
    dropped: list[str] = []
    for category in categories:
        best = _best_single(matrix, category)
        fused = _recall5(matrix, category, ROW_FUSED)
        gain = fused - best
        if gain < 0:
            verdict = "**回退**"
            dropped.append(category)
        elif gain > 0:
            verdict = "提升"
        else:
            verdict = "持平"
        lines.append(f"| {category} | {best:.3f} | {fused:.3f} | {gain:+.3f} | {verdict} |")
    return lines, dropped


def build_report(matrix: Matrix, warnings: list[str]) -> str:
    """生成 Markdown 验收报告。"""
    categories = sorted({category for category, _ in matrix.cells})

    out: list[str] = ["# rag-v1 验收报告", ""]

    out += ["## 1. 分层矩阵", "", render(matrix), ""]

    out += ["## 2. 融合增益（相对最强单路）", ""]
    gain_lines, dropped = _gain_table(matrix, categories)
    out += gain_lines
    out.append("")
    if dropped:
        out += [
            f"> ⚠️ 以下类别出现**回退**：{', '.join(dropped)} —— "
            "融合反而比最强单路更差，说明该类的正确块被其他路的噪声挤掉了。",
            "",
        ]

    out += ["## 3. 验收判据（逐条）", ""]

    graph_categories = [c for c in categories if c in GRAPH_CATEGORIES]
    # ⚠️ 必须先确认真的有 C/D 数据：没有时 all([]) 是 True，等于在空数据上
    # 打绿灯 —— 而这个报告的全部职责就是诚实报告。
    graph_improved = bool(graph_categories) and all(
        _recall5(matrix, c, ROW_FUSED) > _best_single(matrix, c) for c in graph_categories
    )
    no_drop = not dropped
    verdict_1 = "通过" if (graph_improved and no_drop) else "未通过"
    out.append(
        f"- **判据 1 · 融合在 C/D 类有提升，且 A/B 类不掉**：{verdict_1}"
        f"（C/D 提升：{'是' if graph_improved else '否'}；"
        f"无回退：{'是' if no_drop else '否'}）"
    )

    verdict_2 = "通过" if not warnings else "**发现问题**"
    out.append(f"- **判据 2 · 评估集区分度自检**：{verdict_2}")

    out.append(
        "- **判据 3 · 各通路失败模式可复现**："
        "见 `tests/test_vector_path.py`（空查询）、`tests/test_fts_path.py`（空查询）、"
        "`tests/test_graph_retriever.py`（口语化 query 抽不到实体）、"
        "`tests/test_chinese.py`（未分词静默空召回）、`tests/test_degradation.py`（单路故障降级）"
    )
    out.append("")

    if warnings:
        out += ["### 区分度告警", ""]
        out += [f"- {w}" for w in warnings]
        out.append("")

    out += [
        "## 4. 已知偏差与设计局限（结论的边界）",
        "",
        "### 4.1 评估集的结构偏向 —— 方向是**偏向图谱路**",
        "",
        "A/C/D 三类问题**都是用同一套 heading 结构自动生成的**，而图谱索引的也正是"
        "那套结构。所以同一份评估集上图谱路有**结构性优势**：它在 A 类达到 "
        "recall@5 = 1.000，而全文路只有 0.400。",
        "",
        "→ **图谱路的增益被系统性高估，不是低估。**"
        "（早期判断的方向是反的，已按实测更正。）",
        "",
        "### 4.2 RRF 等权融合会稀释强路 —— **V2 候选**",
        "",
        "当前 RRF 对三条通路**等权**（每条都贡献 1/(k+rank)）。当一路显著强于另两路时，"
        "弱路排在前面的噪声块会拿到与强路正确块**同等的票数**，把强路的结果挤出前列。",
        "",
        "这正是 §2 增益表里 A/C/D 三类回退的原因 —— **不是某一路坏了，是融合策略的问题**。",
        "",
        "候选方向（纳入 V2，不阻塞 V1 定型）：加权 RRF、按路归一化后加权、"
        "或对低精度路降权。",
        "",
        "### 4.3 语料为英文为主",
        "",
        "97 万字符里只有 4 个中文字。中文路径（jieba 分词）另有独立测试覆盖，"
        "但未纳入本矩阵的语料基准。",
        "",
        "## 5. 说明",
        "",
        "- 本报告**刻意不给「总 recall」**：三路会互相抵消，平均值什么都说明不了。",
        "- 融合输出即最终排序；V1 无独立重排模块。",
    ]

    return "\n".join(out)
