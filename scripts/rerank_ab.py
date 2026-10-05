"""Rerank A/B：**RRF 直接出 top-k** vs **先召回 top-20 再用 cross-encoder 精排**。

这是 `README.md` 取舍表第 4 行（"V1 无独立重排，cross-encoder 留待 V2"）的
**对照实验**——V2 的第一件事该不该做、做了值多少，这里给数字。

## 协议

- **评估集**：项目自己的 A/B/C/D 四类（`evaluation/qa_gen.py`，按文档结构
  自动生成、零人工标注）。**不是手挑的几条**——手挑的只能证明"我挑的这几条
  变好了"。
- **基线**：`rrf_fuse` 的输出直接取前 k。
- **重排**：取融合后的前 `--depth`（默认 20）块，交给 cross-encoder
  （SiliconFlow `BAAI/bge-reranker-v2-m3`）逐对打分，重排后取前 k。
- 指标：`recall@5` / `recall@10` / `MRR`，**按类别看不看总分**。

## 诚实边界（结果要连着这几条一起讲）

1. **重排只在召回集内重排**：只能重排融合后的前 20，**救不回排在第 21 名之后
   的正确块**。所以它衡量的是**排序质量**，不是召回率——召回率的天花板由
   基线的前 20 决定。
2. **评估集偏向图谱路**（`ACCEPTANCE.md` §4.1）：A/C/D 三类问题与图谱索引
   用的是同一套 heading 结构，图谱路的得分被系统性高估。
3. 单次运行，没有做多次采样——差异小于噪声的类别不要下结论。

用法：

    py scripts/rerank_ab.py                       # 每类 15 条，共 60 条
    py scripts/rerank_ab.py --per-category 30 --depth 20
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，表格里的 ⚠️ 之类字符会直接抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from ragv1 import config  # noqa: E402
from ragv1.api.server import build_retriever  # noqa: E402
from ragv1.embedding import resolve_api_key  # noqa: E402
from ragv1.evaluation.harness import _mrr, _recall  # noqa: E402
from ragv1.evaluation.qa_gen import generate_qas  # noqa: E402
from ragv1.fusion.rrf import rrf_fuse  # noqa: E402
from ragv1.ingest.elements import chunk_elements  # noqa: E402
from ragv1.ingest.loader import load_document  # noqa: E402
from ragv1.retrieve import ALL_PATHS  # noqa: E402
from ragv1.store.fts_store import FtsStore  # noqa: E402
from ragv1.store.graph_store import GraphStore  # noqa: E402

RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def load_chunks(corpus_dir: Path):
    """与 `ingest/build.py` 同一套切分（QA 生成需要 heading_path，而它没落盘）。"""
    from ragv1.config import MAX_CHARS

    chunks = []
    for path in sorted(p for p in corpus_dir.rglob("*") if p.is_file()):
        doc_id = path.relative_to(corpus_dir).as_posix()
        chunks.extend(chunk_elements(load_document(path, doc_id), doc_id, MAX_CHARS))
    return chunks


def rerank(api_key: str, query: str, documents: list[str], top_n: int) -> list[int]:
    """调 SiliconFlow rerank，返回**按相关性降序**的原始下标。"""
    body = json.dumps(
        {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        RERANK_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
        data = json.loads(resp.read())
    ranked = sorted(data["results"], key=lambda r: r["relevance_score"], reverse=True)
    return [r["index"] for r in ranked]


def main() -> int:
    ap = argparse.ArgumentParser(description="Rerank A/B")
    ap.add_argument("--index-dir", default=str(ROOT / ".indexes" / "kb"))
    ap.add_argument("--per-category", type=int, default=15)
    ap.add_argument("--depth", type=int, default=20, help="召回深度（重排的输入规模）")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    api_key = resolve_api_key()

    print(f"索引      : {idx}")
    print(f"评估集    : A/B/C/D 每类 {args.per_category} 条")
    print(f"召回深度  : top-{args.depth} → 重排 → top-5 / top-10")
    print()

    chunks = load_chunks(Path(config.CORPUS_DIR))
    graph = GraphStore(idx / "graph.db")
    fts = FtsStore(idx / "kb.db")
    qas = generate_qas(chunks, graph, per_category=args.per_category)
    retriever = build_retriever(idx)
    print(f"评估集生成: {len(qas)} 条（块 {len(chunks)}）")

    # (类别, 方案) -> {"recall@5": [...], "recall@10": [...], "mrr": [...]}
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}

    started = time.time()
    for i, qa in enumerate(qas, 1):
        fused = rrf_fuse(retriever.retrieve(qa.question, args.depth, ALL_PATHS))
        # 只把前 depth 条交给重排 —— 重排**救不回**这条线之后的块，这正是
        # 本实验的边界，必须和结果一起讲
        base_ids = [h.chunk_id for h in fused][: args.depth]
        docs = [fts.text_of(cid) or "" for cid in base_ids]
        order = rerank(api_key, qa.question, docs, top_n=args.depth)
        reranked_ids = [base_ids[j] for j in order]

        for name, ids in (("基线(RRF)", base_ids), ("+重排", reranked_ids)):
            bucket = acc.setdefault((qa.category, name), {})
            bucket.setdefault("recall@5", []).append(_recall(ids, qa.answer_chunk_ids, 5))
            bucket.setdefault("recall@10", []).append(
                _recall(ids, qa.answer_chunk_ids, 10)
            )
            bucket.setdefault("mrr", []).append(_mrr(ids, qa.answer_chunk_ids))

        if i % 10 == 0:
            print(f"  …已跑 {i}/{len(qas)}  ({time.time() - started:.0f}s)")

    print()
    print("| 类别 | 指标 | 基线 RRF | +重排 | 增益 |")
    print("|---|---|---|---|---|")
    order = ["A", "B", "C", "D"]
    for cat in order + ["全体"]:
        cats = order if cat == "全体" else [cat]
        for metric in ("recall@5", "recall@10", "mrr"):
            base = _agg(cats, "基线(RRF)", metric, acc)
            rr = _agg(cats, "+重排", metric, acc)
            print(f"| {cat} | {metric} | {base:.3f} | {rr:.3f} | {rr - base:+.3f} |")
    print()
    print("⚠️ 重排只在召回集内重排（救不回第 21 名之后的正确块）；")
    print("   评估集偏向图谱路（ACCEPTANCE.md §4.1）；单次运行，小差异不下结论。")
    return 0


def _agg(cats: list[str], name: str, metric: str, acc) -> float:
    values: list[float] = []
    for cat in cats:
        values.extend(acc.get((cat, name), {}).get(metric, []))
    return sum(values) / len(values) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
