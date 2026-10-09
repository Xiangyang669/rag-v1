"""V2 验收脚本 —— **检索表部分**（任务 8 新建；任务 18 在此基础上扩答案表）。

判据 1：**加权融合的 `recall@5` ≥ 最强单路**。V1 的等权 RRF 没通过这条判据，
本脚本在给定评估集上打印照表把「加权之后赢没赢」变成数字：

    三条单路（vector / fulltext / graph）
    fused_eq     —— 等权 RRF（V1 基线）
    fused_static —— 加权 RRF，权重取 `config.STATIC_FUSION_WEIGHTS`（任务 8 定档）
    fused_routed —— 查询自适应加权（`route_weights`，任务 6/7 的生产默认）

## R15：检索指标排除 `unanswerable`

`harness._recall` 对空答案集按设计返回 0.0（`harness.py:39-42`）。评估集里
`category == "unanswerable"` 的题 `answer_chunk_ids` 为空，若不排除会把三条
单路与融合**同时**拉向 0，让「融合是否胜过最强单路」的比较失去意义。故
`retrieval_items()` 显式过滤它们——这些题只由拒答指标（任务 15）来评。

## 每条查询只检索一次

同 `tune_weights`：`retrieve` 每调一次走一次 embedding。这里每条题只
`retrieve` 一次，之后的六行全部在内存里融合。

用法：

    py scripts/eval_v2.py --index-dir .indexes/kb_zh \
        --set ragv1/evaluation/qa_sets/zh_test.jsonl --retrieval-only
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，表格里的中文/符号会直接抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from ragv1.config import STATIC_FUSION_WEIGHTS  # noqa: E402
from ragv1.evaluation.harness import _mrr, _recall  # noqa: E402
from ragv1.evaluation.qaset import load_qaset  # noqa: E402
from ragv1.fusion.rrf import rrf_fuse  # noqa: E402
from ragv1.fusion.router import route_weights  # noqa: E402
from ragv1.retrieve import ALL_PATHS  # noqa: E402

UNANSWERABLE = "unanswerable"

# 行名与顺序。单路在前，融合在后。
ROW_VECTOR = "vector"
ROW_FULLTEXT = "fulltext"
ROW_GRAPH = "graph"
ROW_FUSED_EQ = "fused_eq"
ROW_FUSED_STATIC = "fused_static"
ROW_FUSED_ROUTED = "fused_routed"
ROW_ORDER = (
    ROW_VECTOR,
    ROW_FULLTEXT,
    ROW_GRAPH,
    ROW_FUSED_EQ,
    ROW_FUSED_STATIC,
    ROW_FUSED_ROUTED,
)
SINGLE_ROWS = (ROW_VECTOR, ROW_FULLTEXT, ROW_GRAPH)


def retrieval_items(items):
    """R15：剔掉 unanswerable（空答案集按设计 recall=0，会污染单路 vs 融合的比较）。"""
    return tuple(it for it in items if it.category != UNANSWERABLE)


def _row_order(names):
    """已知行名按 ROW_ORDER 顺序，其余按名字排序——保证表格可复现。"""
    return sorted(names, key=lambda n: (ROW_ORDER.index(n) if n in ROW_ORDER else 99, n))


def _ids(seq):
    """从 Hit / FusedHit 序列取 chunk_id（两者都有该属性）。"""
    return [h.chunk_id for h in seq]


def retrieval_table(rows, answer_ids):
    """把若干「行」（`{行名: [Hit|FusedHit, ...]}`）渲染成指标表。

    ⚠️ 计划原文（R7）把任务 18 的调用写成 `retrieval_table(["a"], rows)`——
    两个参数写反了。这里容忍两种顺序，免得下游照原文调用时炸。
    """
    if isinstance(rows, (list, tuple)) and isinstance(answer_ids, dict):
        rows, answer_ids = answer_ids, rows
    answers = tuple(answer_ids)

    lines = ["| 行 | recall@5 | recall@10 | MRR |", "|---|---|---|---|"]
    for name in _row_order(rows):
        ids = _ids(rows[name])
        lines.append(
            f"| {name} | {_recall(ids, answers, 5):.3f} | "
            f"{_recall(ids, answers, 10):.3f} | {_mrr(ids, answers):.3f} |"
        )
    return "\n".join(lines)


def row_hits(ranked, query, weights):
    """一条查询的六行结果：三条单路 + 等权 / 静态加权 / 路由加权融合。"""
    rows = {path: ranked.get(path, []) for path in SINGLE_ROWS}
    rows[ROW_FUSED_EQ] = rrf_fuse(ranked, weights=None)
    rows[ROW_FUSED_STATIC] = rrf_fuse(ranked, weights=weights)
    rows[ROW_FUSED_ROUTED] = rrf_fuse(ranked, weights=route_weights(query, ranked))
    return rows


def evaluate(retriever, items, weights, k_values=(5, 10)):
    """逐题检索一次、算六行指标，按行求平均。返回 `{行: {指标: 均值}}`。"""
    depth = max(k_values)
    acc: dict[str, dict[str, list[float]]] = {}

    for it in items:
        # ── 唯一一次检索：每条题、三条路各取一遍 ──
        ranked = retriever.retrieve(it.question, depth, ALL_PATHS)
        for name, seq in row_hits(ranked, it.question, weights).items():
            ids = _ids(seq)[:depth]
            bucket = acc.setdefault(
                name, {"recall@5": [], "recall@10": [], "mrr": []}
            )
            bucket["recall@5"].append(_recall(ids, it.answer_chunk_ids, 5))
            bucket["recall@10"].append(_recall(ids, it.answer_chunk_ids, 10))
            bucket["mrr"].append(_mrr(ids, it.answer_chunk_ids))

    return {
        name: {metric: sum(v) / len(v) for metric, v in bucket.items()}
        for name, bucket in acc.items()
    }


def render_table(metrics):
    """把 `{行: {指标: 均值}}` 渲染成 Markdown（行名按 ROW_ORDER）。"""
    lines = ["| 行 | recall@5 | recall@10 | MRR |", "|---|---|---|---|"]
    for name in _row_order(metrics):
        cell = metrics[name]
        lines.append(
            f"| {name} | {cell['recall@5']:.3f} | {cell['recall@10']:.3f} "
            f"| {cell['mrr']:.3f} |"
        )
    return "\n".join(lines)


def criterion_1(metrics, fused_row=ROW_FUSED_STATIC):
    """判据 1：加权融合 recall@5 ≥ 最强单路。返回 (是否通过, 文案)。"""
    fused = metrics[fused_row]["recall@5"]
    best_name = max(SINGLE_ROWS, key=lambda n: metrics[n]["recall@5"])
    best = metrics[best_name]["recall@5"]
    ok = fused >= best
    verdict = "通过 ✅" if ok else "未通过 ❌"
    return ok, (
        f"判据 1（{fused_row} vs 最强单路）: recall@5 {fused:.3f} vs "
        f"{best:.3f}（{best_name}） → {verdict}"
    )


def _counting_retriever(index_dir):
    """装配真实 retriever，并包一层 embed_fn 数 API 往返次数（不读、不打印 key）。"""
    from ragv1.api.server import build_retriever
    from ragv1.embedding import default_embed_fn

    calls = {"n": 0}
    real = default_embed_fn()

    def counting_embed(texts):
        calls["n"] += 1
        return real(texts)

    return build_retriever(index_dir, embed_fn=counting_embed), calls


def main() -> int:
    ap = argparse.ArgumentParser(description="V2 验收（检索表）")
    ap.add_argument("--index-dir", default=str(ROOT / ".indexes" / "kb_zh"))
    ap.add_argument(
        "--set",
        default=str(ROOT / "ragv1" / "evaluation" / "qa_sets" / "zh_test.jsonl"),
        help="评估集（留出集只用一次；调参请用 tune_weights 的 dev 集）",
    )
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只跑检索表（答案表由任务 18 扩展）",
    )
    args = ap.parse_args()

    items = retrieval_items(load_qaset(args.set))
    retriever, calls = _counting_retriever(args.index_dir)

    print(f"索引      : {args.index_dir}")
    print(f"评估集    : {args.set}（排除 unanswerable 后 {len(items)} 条）")
    print(f"静态权重  : {STATIC_FUSION_WEIGHTS}")
    print()

    metrics = evaluate(retriever, items, STATIC_FUSION_WEIGHTS, (5, 10))
    print(render_table(metrics))
    print()

    _, line_static = criterion_1(metrics, ROW_FUSED_STATIC)
    _, line_routed = criterion_1(metrics, ROW_FUSED_ROUTED)
    print(line_static)
    print(line_routed)
    print(
        "等权基线（V1 未通过的形态）: "
        f"fused_eq recall@5 = {metrics[ROW_FUSED_EQ]['recall@5']:.3f}"
    )
    print(f"检索次数（embedding API 往返）: {calls['n']} = {len(items)} 条 × 1 次/条")

    if not args.retrieval_only:
        print()
        print("（未传 --retrieval-only：答案表 / --with-support 由任务 18 扩展）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
