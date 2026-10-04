"""入库验收：对真实语料跑一遍，把实测数字打出来。

⚠️ 索引写在 `.indexes/verify_ingest/`（已被 .gitignore 忽略），**不碰**
你的 `.indexes/` 主索引。脚本每次先清空这个子目录再重建，保证数字可复现。

⚠️ **会真的下载语料里的远程图片**（实测 184 处，全在 raw.githubusercontent.com），
并按配置真跑 OCR / 多模态。首次很慢，之后图片走本地缓存。

对照基线（spec §1，改造前实测）：
    表格 117 张（围栏感知口径）；超 1200 字符的会被切成多块

用法：
    py scripts/verify_ingest.py                  # 全量（含图片通道，网络慢时很久）
    py scripts/verify_ingest.py --no-images      # 只验正文与表格，很快
    py scripts/verify_ingest.py --fetch-budget 12  # 图片通道，最多取图 12 次

⚠️ **网络会拖死全量跑**：实测本机到 raw.githubusercontent.com 每次取图要
40-68 秒且大多失败，184 张图 = 2.5 小时以上。所以给了 `--fetch-budget`
来把「图片通道 + 降级路径」在有限时间内验完——取图预算用尽后，后续图片
一律落 `image_fetch_failed`，那正是断网生产环境下的真实行为。
"""

import argparse
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import fake_embed  # noqa: E402 —— 与单测用同一个确定性假向量

from ragv1 import config  # noqa: E402
from ragv1.ingest import table  # noqa: E402
from ragv1.ingest.build import build_corpus  # noqa: E402
from ragv1.ingest.ocr import build_ocr_engine  # noqa: E402
from ragv1.ingest.vlm import build_vision_engine  # noqa: E402
from ragv1.store.fts_store import FtsStore  # noqa: E402

REPORT = ROOT / "verify_report.txt"
INDEX_DIR = ROOT / ".indexes" / "verify_ingest"

# 围栏感知口径的表格基线（见 .superpowers 里的裁决：152 是朴素 grep 的误算）
BASELINE_TABLES = 117


def _budgeted_fetcher(budget: int):
    """限预算的取图器：调用超过 budget 次后一律失败。

    用来把「取图失败 → 降级不中断」这条路径在有限时间内验完——而不是被
    慢网络拖成几小时。
    """
    import urllib.request

    calls = {"n": 0}

    def fetch(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] > budget:
            raise OSError(f"取图预算已用尽（{budget} 次）")
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            return resp.read()

    return fetch


def _measure(index_dir: Path, ocr, vlm, fetcher=None):
    """跑一次入库并把要报的数字收成纯数据。"""
    started = time.time()
    total = build_corpus(
        config.CORPUS_DIR,
        index_dir,
        embed_fn=fake_embed,
        ocr=ocr,
        vlm=vlm,
        fetcher=fetcher,
    )
    elapsed = time.time() - started

    store = FtsStore(index_dir / "kb.db")
    kinds = Counter()
    tables = []
    images = []
    degrade_hits = Counter()
    pages = orders = 0

    for cid in store.chunk_ids():
        meta = store.meta_of(cid)
        text = store.text_of(cid) or ""
        kinds[meta["kind"]] += 1
        if meta["kind"] == "table":
            tables.append((cid, meta, text))
        elif meta["kind"] == "image":
            images.append((cid, meta, text))
        for code in meta["degrade"]:
            degrade_hits[code] += 1
        pages += meta["page"] is not None
        orders += meta["order"] is not None

    return total, elapsed, kinds, tables, images, degrade_hits, pages, orders


def main() -> int:
    parser = argparse.ArgumentParser(description="入库验收（见文件头的用法）")
    parser.add_argument(
        "--no-images", action="store_true", help="不做图片解析（图片按正文处理），很快"
    )
    parser.add_argument(
        "--fetch-budget", type=int, default=0, help="最多取图多少次（0 = 不限）"
    )
    args = parser.parse_args()

    if not config.CORPUS_DIR.exists():
        print(f"[跳过] 语料目录不存在：{config.CORPUS_DIR}")
        return 1

    files = [p for p in config.CORPUS_DIR.rglob("*") if p.is_file()]
    by_suffix = Counter(p.suffix.lower() for p in files)

    ocr = None if args.no_images else build_ocr_engine(config.OCR_ENGINE)
    vlm = None if args.no_images else build_vision_engine(config.VLM_ENGINE)
    fetcher = _budgeted_fetcher(args.fetch_budget) if args.fetch_budget > 0 else None
    print(
        f"引擎：OCR={config.OCR_ENGINE}({'就绪' if ocr else '关闭'})  "
        f"VLM={config.VLM_ENGINE}({'就绪' if vlm else '关闭'})  "
        f"取图预算={args.fetch_budget or '不限'}",
        flush=True,
    )

    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (total, elapsed, kinds, tables, images, degrade_hits, pages, orders) = _measure(
        INDEX_DIR, ocr, vlm, fetcher
    )

    # ── 表格 ──
    table_lens = [len(text) for _, _, text in tables]
    over_cap = [n for n in table_lens if n > config.MAX_CHARS]
    split_tables = [m for _, m, _ in tables if m["part"] > 0]
    distinct_tables = len(tables) - len(split_tables)

    # 行完整性：表格部分（第一条表行起）的每一行都必须以 | 收尾。
    # 不假定「第 0 行是 heading 前缀」——PDF 出来的表就没有前缀。
    broken_rows = []
    for cid, _, text in tables:
        rows = text.splitlines()
        start = next((i for i, ln in enumerate(rows) if table.is_table_line(ln)), None)
        if start is None:
            broken_rows.append((cid, rows[0] if rows else "<空块>"))
            continue
        broken_rows.extend(
            (cid, ln)
            for ln in rows[start:]
            if not (ln.strip().startswith("|") and ln.strip().endswith("|"))
        )
    data_rows = sum(
        max(len([ln for ln in text.splitlines() if table.is_table_line(ln)]) - 2, 0)
        for _, _, text in tables
    )

    # ── 图片 ──
    with_ocr = sum(1 for _, _, t in images if "文字:" in t)
    with_vlm = sum(1 for _, _, t in images if "类型:" in t)
    with_both = sum(1 for _, _, t in images if "文字:" in t and "类型:" in t)
    with_path = sum(1 for _, m, _ in images if m["image_path"])
    with_ref = sum(1 for _, m, _ in images if m["image_ref"])

    lines = [
        "=== 入库验收报告 ===",
        f"语料目录：{config.CORPUS_DIR}",
        f"文件总数：{len(files)}  按后缀：{dict(by_suffix)}",
        f"入库耗时：{elapsed:.1f} 秒",
        "",
        f"写入块数：{total}",
        f"kind 分布：{dict(kinds)}",
        f"带页码的块：{pages}   带阅读序号的块：{orders} / {total}",
        "",
        "── 表格 ──",
        f"表格块数：{len(tables)}（其中 part>0 的 {len(split_tables)} 块）",
        f"→ 去重后的表格张数：{distinct_tables}   （围栏感知基线：{BASELINE_TABLES}）",
        f"表格数据行总数：{data_rows}",
        f"表格块最长字符数：{max(table_lens, default=0)}（MAX_CHARS={config.MAX_CHARS}）",
        f"超过 MAX_CHARS 的表格块：{len(over_cap)}"
        "  ← 只有「单行本身就超预算」才会 >0",
        "",
        "── 图片 ──",
        f"图片块数：{len(images)}",
        f"  带原始引用的：{with_ref}      带本地路径的：{with_path}",
        f"  OCR 贡献了文字的：{with_ocr}",
        f"  多模态贡献了描述的：{with_vlm}",
        f"  **两条通道都贡献的：{with_both}**",
        "",
        f"降级码分布：{dict(degrade_hits) or '（无）'}",
        "",
        "判定：",
        f"  [{'OK' if distinct_tables >= BASELINE_TABLES else '!!'}] "
        f"表格张数 {distinct_tables} ≥ 基线 {BASELINE_TABLES}",
        f"  [{'OK' if split_tables else '!!'}] 有大表被切成多块（{len(split_tables)} 块 part>0）",
        f"  [{'OK' if not broken_rows else '!!'}] 表格行完整性："
        f"{'没有任何一行被切断' if not broken_rows else f'{len(broken_rows)} 行被切断！'}",
        f"  [{'OK' if len(images) > 0 else '!!'}] 图片被识别成独立元素（{len(images)} 块）",
        f"  [{'OK' if with_path == len(images) or not images else '!!'}] "
        f"图片块都带本地路径（{with_path}/{len(images)}）",
        "",
        "被跳过的文件类型（load_document 不认，返回空列表）：",
        "  " + ", ".join(
            f"{s}×{n}"
            for s, n in sorted(by_suffix.items())
            if s not in {".md", ".markdown", ".pdf"}
        ),
    ]
    for cid, line in broken_rows[:5]:
        lines.append(f"  !! 残缺行 {cid}: {line[:60]}")

    report = "\n".join(lines) + "\n"
    REPORT.write_text(report, encoding="utf-8")
    print(report.encode("ascii", "replace").decode("ascii"))
    print(f"完整报告（UTF-8）：{REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
