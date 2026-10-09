"""dev 集上的静态融合权重网格搜索（任务 8）。

V1 的**等权** RRF 在 A/C/D 类问题上不如最强单路（RRF 等权稀释：弱路把强路
的优势拉平）。加权之后能否翻盘，要用数字说话——本脚本在 dev 集上把
(vector, fulltext) 的权重网格搜一遍，按 `recall@5` 排序，选出的最优静态基线
写回 `config.STATIC_FUSION_WEIGHTS`。**在 dev 上调参、在 test 上报数**：本脚本
只碰 dev 集，test 集（`zh_test.jsonl`）由 `scripts/eval_v2.py` 评一次。

## 关键工程约束：每条查询只检索一次

`Retriever.retrieve` 每调一次就走一次 embedding（真实 API 往返）。若每个权重
组合都重新检索一遍，dev 18 条 × 231 组合 = 四千多次往返，既慢又烧钱。所以
本脚本先把每条题的 `ranked` **取一次并缓存**，再在内存里对每个组合调
`rrf_fuse(ranked, weights=combo)`。检索只发生在 `grid()` 构建缓存那一次，
之后的组合评估是纯内存融合（见 `grid` / `score_weights`）。

## R15：检索指标排除 `unanswerable`

`harness._recall` 对空答案集按设计返回 0.0（`harness.py:39-42`）。评估集里
`category == "unanswerable"` 的题 `answer_chunk_ids` 为空，若不排除会把三条
单路与融合**同时**拉向 0，让「融合是否胜过最强单路」的比较失去意义，还会把
权重搜到噪声上。故 `retrieval_items()`（本文件）显式过滤它们——这些题只由
拒答指标（任务 15）来评。

用法：

    py scripts/tune_weights.py --index-dir .indexes/kb_zh \
        --dev ragv1/evaluation/qa_sets/zh_dev.jsonl
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

from ragv1.evaluation.harness import _mrr, _recall  # noqa: E402
from ragv1.evaluation.qaset import load_qaset  # noqa: E402
from ragv1.fusion.rrf import rrf_fuse  # noqa: E402
from ragv1.retrieve import ALL_PATHS  # noqa: E402

UNANSWERABLE = "unanswerable"

# 融合涉及的三条路。`STATIC_FUSION_WEIGHTS` 是 `route_weights` 的基线——
# 若某路基线为 0，路由的 boost（×1.8）乘 0 也永远救不活它，三路检索就退化成
# 单路。故上机取值要求三路权重都不低于此下限（全局最优落在角点时另行报告）。
PATHS = ("vector", "fulltext", "graph")
MIN_PATH_WEIGHT = 0.05

# 网格搜索的两条坐标轴：vector / fulltext 各 0–1、步长 0.05。
# graph = 1 − vector − fulltext；graph < 0 的组合在 weight_combos 里被跳过，
# 所以有效网格就是三单纯形 {v,f,g ≥ 0, v+f+g = 1} 上的 231 个格点。
_AXIS = tuple(round(i * 0.05, 2) for i in range(21))
VECTOR_AXIS = _AXIS
FULLTEXT_AXIS = _AXIS


def retrieval_items(items):
    """R15：剔掉 unanswerable（空答案集按设计 recall=0，会污染单路 vs 融合的比较）。"""
    return tuple(it for it in items if it.category != UNANSWERABLE)


def weight_combos(*axes):
    """(vector, fulltext) 两轴的笛卡尔积 → 归一化的权重字典列表。

    `graph = 1 − vector − fulltext`；`graph < 0` 的组合**跳过**（不允许负权重）。
    返回顺序固定（先 vector、后 fulltext 的嵌套遍历）——保证可复现。
    """
    if len(axes) != 2:
        raise ValueError("weight_combos 需要两条轴：(vector_values, fulltext_values)")
    vector_values, fulltext_values = axes
    combos: list[dict[str, float]] = []
    for v in vector_values:
        for f in fulltext_values:
            g = 1.0 - v - f
            if g < -1e-9:  # 容忍浮点噪声；真正为负的组合才是负权重
                continue
            combos.append({"vector": v, "fulltext": f, "graph": max(g, 0.0)})
    return combos


def score_weights(ranked_by_question, items, weights, k_values=(5, 10)):
    """在**已检索缓存**上评一组权重，返回 `{recall@k..., mrr}`。

    `ranked_by_question[题]` 是题目的三路原始 ranked（由 `grid` 一次性取好）。
    ⚠️ 这里刻意**不接收 retriever**——从签名上就杜绝了「每个组合重检索一遍」；
    要重检索，请走 `grid()` 一次构建缓存。

    指标复用 `harness._recall` / `_mrr`（不另写一套：两处实现漂移会让新旧数字
    不可比）。
    """
    largest = max(k_values)
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    mrrs: list[float] = []
    for it in items:
        fused = rrf_fuse(ranked_by_question[it.question], weights=weights)
        ids = [h.chunk_id for h in fused[:largest]]
        for k in k_values:
            recalls[k].append(_recall(ids, it.answer_chunk_ids, k))
        mrrs.append(_mrr(ids, it.answer_chunk_ids))

    if not items:  # 空集：不猜，明确返回 0
        return {**{f"recall@{k}": 0.0 for k in k_values}, "mrr": 0.0}

    out = {f"recall@{k}": sum(v) / len(v) for k, v in recalls.items()}
    out["mrr"] = sum(mrrs) / len(mrrs)
    return out


def rank_results(results):
    """按 `recall@5` 降序排；平局用**权重字典的 `sorted(items())`** 稳定打破。

    `sorted(dict.items())` 是 (键, 值) 元组列表，元素可比且顺序确定——不依赖
    dict 插入顺序以外的偶然因素，同一输入两次运行结果逐字节一致。
    """
    return sorted(
        results,
        key=lambda pair: (-pair[1]["recall@5"], sorted(pair[0].items())),
    )


def best_all_path(ranked, min_weight=MIN_PATH_WEIGHT):
    """在「三路权重都 ≥ min_weight」的子集里取 recall@5 最优（同 rank_results 排序）。

    为什么需要它：全局最优常常落在**角点**（把某一路权重压成 0），那实质是
    "关掉几路"而非"融合"。作为 `route_weights` 的静态基线，必须让三路都留有
    非零权重，否则路由的自适应调整全部失效（乘 0 永远是 0）。角点最优另行报告。
    """
    allowed = [
        r for r in ranked if all(r[0].get(p, 0.0) >= min_weight for p in PATHS)
    ]
    return rank_results(allowed)[0] if allowed else ranked[0]


def grid(retriever, items, k_values, combos, paths=ALL_PATHS):
    """跑网格：**每条题只检索一次**，然后每个组合在内存里融合。

    返回 `rank_results` 的输出：`[({权重}, {指标}), ...]`，按 recall@5 降序。
    """
    depth = max(k_values)
    # ── 唯一一次检索：每条题、三条路各取一遍 ──
    cache = {
        it.question: retriever.retrieve(it.question, depth, paths) for it in items
    }
    results = [(w, score_weights(cache, items, w, k_values)) for w in combos]
    return rank_results(results)


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
    ap = argparse.ArgumentParser(description="dev 集静态融合权重网格搜索")
    ap.add_argument(
        "--index-dir",
        default=str(ROOT / ".indexes" / "kb_zh"),
        help="中文索引目录",
    )
    ap.add_argument(
        "--dev",
        default=str(ROOT / "ragv1" / "evaluation" / "qa_sets" / "zh_dev.jsonl"),
        help="调参用评估集（**不许碰 test 集**）",
    )
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    items = retrieval_items(load_qaset(args.dev))
    k_values = tuple(args.k)
    combos = weight_combos(VECTOR_AXIS, FULLTEXT_AXIS)

    retriever, calls = _counting_retriever(args.index_dir)

    print(f"索引      : {args.index_dir}")
    print(f"dev 集    : {args.dev}（排除 unanswerable 后 {len(items)} 条）")
    print(
        f"搜索范围  : vector {VECTOR_AXIS[0]:.2f}–{VECTOR_AXIS[-1]:.2f}、"
        f"fulltext {FULLTEXT_AXIS[0]:.2f}–{FULLTEXT_AXIS[-1]:.2f}、步长 0.05"
        f"（graph = 1−v−f ≥ 0）→ {len(combos)} 个组合"
    )
    print(f"检索深度  : top-{max(k_values)}")
    print()

    ranked = grid(retriever, items, k_values, combos)

    print(f"| 排名 | vector | fulltext | graph | recall@5 | recall@10 | MRR |")
    print("|---|---|---|---|---|---|---|")
    for i, (w, m) in enumerate(ranked[: args.top], 1):
        print(
            f"| {i} | {w['vector']:.2f} | {w['fulltext']:.2f} | {w['graph']:.2f} "
            f"| {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} |"
        )
    print()

    best_w, best_m = ranked[0]
    ship_w, ship_m = best_all_path(ranked)
    print(
        "全局最优（角点，可能把某路压成 0）: "
        f"{{'vector': {best_w['vector']:.2f}, 'fulltext': {best_w['fulltext']:.2f}, "
        f"'graph': {best_w['graph']:.2f}}}  →  "
        f"recall@5={best_m['recall@5']:.3f}, MRR={best_m['mrr']:.3f}"
    )
    print(
        f"上机建议（三路权重均 ≥ {MIN_PATH_WEIGHT}，对齐 route_weights 基线）: "
        f"{{'vector': {ship_w['vector']:.2f}, 'fulltext': {ship_w['fulltext']:.2f}, "
        f"'graph': {ship_w['graph']:.2f}}}  →  "
        f"recall@5={ship_m['recall@5']:.3f}, MRR={ship_m['mrr']:.3f}"
    )
    print(f"检索次数（embedding API 往返）: {calls['n']} = {len(items)} 条 × 1 次/条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
