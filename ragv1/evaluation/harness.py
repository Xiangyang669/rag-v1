"""分层指标 harness：**四类 × 四路**矩阵。

⚠️ 刻意**不提供任何"总 recall"聚合接口**——三路会互相抵消（向量赢的地方
图谱输），一个平均值什么都说明不了。要看的结论形态是分类型的差异：

    「融合后在 C/D 类上把 recall@5 从 0.2 拉到 0.75，而 A/B 类不掉。」

主指标 recall@k（RAG 的瓶颈在召回，不在排序），辅指标 MRR。
"""

from dataclasses import dataclass, field

from ragv1.evaluation.qa_gen import QAItem
from ragv1.fusion.rrf import rrf_fuse
from ragv1.retrieve import ALL_PATHS

ROW_VECTOR = "vector"
ROW_FULLTEXT = "fulltext"
ROW_GRAPH = "graph"
ROW_FUSED = "fused"

ROWS: tuple[str, ...] = (ROW_VECTOR, ROW_FULLTEXT, ROW_GRAPH, ROW_FUSED)

METRIC_MRR = "mrr"


@dataclass(frozen=True)
class EvalConfig:
    k_values: tuple[int, ...] = (5, 10)


@dataclass
class Matrix:
    """cells[(类别, 通路)] = {指标名: 值}"""

    cells: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)


def _recall(hits: list[str], answers: tuple[str, ...], k: int) -> float:
    if not answers:
        return 0.0
    return len(set(hits[:k]) & set(answers)) / len(set(answers))


def _mrr(hits: list[str], answers: tuple[str, ...]) -> float:
    answer_set = set(answers)
    for rank, chunk_id in enumerate(hits, start=1):
        if chunk_id in answer_set:
            return 1.0 / rank
    return 0.0


def _ranked_ids(retriever, query: str, row: str, k: int) -> list[str]:
    """取出某一行（单路或融合）的块 ID 排序。

    融合行用 ALL_PATHS 跑一遍再 rrf_fuse —— 它衡量的是"三路合起来"的效果，
    而不是某一单路。
    """
    if row == ROW_FUSED:
        return [h.chunk_id for h in rrf_fuse(retriever.retrieve(query, k, paths=ALL_PATHS))]

    per_path = retriever.retrieve(query, k, paths={row})
    return [h.chunk_id for h in per_path.get(row, [])]


def evaluate(retriever, qas: list[QAItem], cfg: EvalConfig) -> Matrix:
    """按 (类别, 通路) 聚合 recall@k 与 MRR。"""
    largest_k = max(cfg.k_values)
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}

    for qa in qas:
        for row in ROWS:
            hits = _ranked_ids(retriever, qa.question, row, largest_k)
            bucket = acc.setdefault((qa.category, row), {})
            for k in cfg.k_values:
                bucket.setdefault(f"recall@{k}", []).append(
                    _recall(hits, qa.answer_chunk_ids, k)
                )
            bucket.setdefault(METRIC_MRR, []).append(_mrr(hits, qa.answer_chunk_ids))

    cells = {
        key: {name: sum(values) / len(values) for name, values in bucket.items()}
        for key, bucket in acc.items()
    }
    return Matrix(cells=cells)


def _metric_order(cell: dict[str, float]) -> list[str]:
    recalls = sorted(
        (m for m in cell if m.startswith("recall@")),
        key=lambda m: int(m.split("@", 1)[1]),
    )
    return recalls + ([METRIC_MRR] if METRIC_MRR in cell else [])


def render(matrix: Matrix) -> str:
    """渲染成可读的 Markdown 表格。"""
    categories = sorted({category for category, _ in matrix.cells})
    sample = next(iter(matrix.cells.values()), {})
    metrics = _metric_order(sample)

    lines = [
        "| 类别 | 通路 | " + " | ".join(metrics) + " |",
        "|---|---|" + "---|" * len(metrics),
    ]
    for category in categories:
        for row in ROWS:
            cell = matrix.cells.get((category, row), {})
            values = " | ".join(f"{cell.get(m, 0.0):.3f}" for m in metrics)
            lines.append(f"| {category} | {row} | {values} |")
    return "\n".join(lines)


def check_discrimination(matrix: Matrix, threshold: float = 0.05) -> list[str]:
    """标出**测不出差异**的类别。

    对每个类，取三条单路 recall@5 的最大值与最小值——差值小于阈值即告警。
    这种类乘上的问题，三路答得一样好（或一样差），保留它只会稀释结论。

    注意只看**单路**，不含 fused：融合行的意义是"比单路好"，把它算进
    离散度会把真正的信号抹平。
    """
    single_paths = (ROW_VECTOR, ROW_FULLTEXT, ROW_GRAPH)
    warnings: list[str] = []

    for category in sorted({c for c, _ in matrix.cells}):
        values = [
            matrix.cells.get((category, row), {}).get("recall@5") for row in single_paths
        ]
        present = [v for v in values if v is not None]
        if len(present) < 2:
            continue  # 只有一路有数据时谈不上"差异"

        spread = max(present) - min(present)
        if spread < threshold:
            warnings.append(
                f"类别 {category} 无区分力：三条单路 recall@5 差值仅 {spread:.3f}"
                f"（阈值 {threshold}），该类需加强或删除"
            )

    return warnings
