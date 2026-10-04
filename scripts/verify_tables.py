"""阶段一验收：对真实语料跑一遍入库，把实测数字打出来。

⚠️ 索引写在 `.indexes/verify_tables/`（已被 .gitignore 忽略），**不碰**
你的 `.indexes/` 主索引。脚本每次先清空这个子目录再重建，保证数字可复现。

对照基线（spec §1，改造前实测）：
    表格 152 张；超 1200 字符的 9 张；最长表格 3582 字符

用法：
    py scripts/verify_tables.py            # 打印摘要 + 写 verify_report.txt
"""

import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import fake_embed  # noqa: E402 —— 与单测用同一个确定性假向量

from ragv1.config import CORPUS_DIR, MAX_CHARS  # noqa: E402
from ragv1.ingest.build import build_corpus  # noqa: E402
from ragv1.store.fts_store import FtsStore  # noqa: E402

REPORT = ROOT / "verify_report.txt"
INDEX_DIR = ROOT / ".indexes" / "verify_tables"

# 改造前的建表基线。
#
# ⚠️ spec §1 里写的 152 是**朴素 grep** 的结果（只数「连续以 | 开头的行」），
# 它把两类东西误算成了表格：代码围栏里的 | 行（实测 62 处）、以及第 2 行不是
# 分隔行的伪表。围栏感知后的真实表格数是 117 —— 这也是本次入库存出的
# 「表格数」（表格块数减去被切分的块数）。
BASELINE_TABLES = 117


def _measure(index_dir: Path) -> tuple[int, Counter, list, Counter]:
    """跑一次入库并把要报的数字收成纯数据。"""
    total = build_corpus(CORPUS_DIR, index_dir, embed_fn=fake_embed)

    store = FtsStore(index_dir / "kb.db")
    kinds: Counter = Counter()
    tables: list = []
    degrade_hits: Counter = Counter()
    for cid in store.chunk_ids():
        meta = store.meta_of(cid)
        kinds[meta["kind"]] += 1
        if meta["kind"] == "table":
            tables.append((cid, meta, store.text_of(cid)))
        for code in meta["degrade"]:
            degrade_hits[code] += 1
    return total, kinds, tables, degrade_hits


def main() -> int:
    if not CORPUS_DIR.exists():
        print(f"[跳过] 语料目录不存在：{CORPUS_DIR}")
        return 1

    files = [p for p in CORPUS_DIR.rglob("*") if p.is_file()]
    by_suffix = Counter(p.suffix.lower() for p in files)

    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    total, kinds, tables, degrade_hits = _measure(INDEX_DIR)

    table_lens = [len(text) for _, _, text in tables]
    over_cap = [n for n in table_lens if n > MAX_CHARS]
    split_tables = [m for _, m, _ in tables if m["part"] > 0]
    distinct_tables = len(tables) - len(split_tables)

    # 核心证据：每个表格块里的每一行是否都是**完整**的表行。
    # 只要有任意一行不以 | 开头/结尾收尾，就说明它被从中间切断了。
    # （第一行是 heading_path 前缀，不是表行，跳过。）
    broken_rows = [
        (cid, line)
        for cid, _, text in tables
        for line in text.splitlines()[1:]
        if not (line.strip().startswith("|") and line.strip().endswith("|"))
    ]
    data_rows = sum(len(text.splitlines()) - 3 for _, _, text in tables)

    lines = [
        "=== 阶段一验收报告（表格） ===",
        f"语料目录：{CORPUS_DIR}",
        f"文件总数：{len(files)}  按后缀：{dict(by_suffix)}",
        "",
        f"写入块数：{total}",
        f"kind 分布：{dict(kinds)}",
        "",
        f"表格块数：{len(tables)}",
        f"其中被切分的块（part>0）：{len(split_tables)}",
        f"→ 去重后的表格张数：{distinct_tables}   （围栏感知基线：{BASELINE_TABLES}）",
        f"表格数据行总数：{data_rows}",
        f"表格块最长字符数：{max(table_lens, default=0)}（MAX_CHARS={MAX_CHARS}）",
        f"超过 MAX_CHARS 的表格块：{len(over_cap)}"
        f"  ← 只有「单行本身就超预算」才会 >0，属预期内的降级",
        "",
        f"降级码分布：{dict(degrade_hits) or '（无）'}",
        "",
        "判定：",
        f"  [{'OK' if distinct_tables >= BASELINE_TABLES else '!!'}] "
        f"表格张数 {distinct_tables} ≥ 基线 {BASELINE_TABLES}",
        f"  [{'OK' if split_tables else '!!'}] 有大表被切成多块（{len(split_tables)} 块 part>0）",
        f"  [{'OK' if not broken_rows else '!!'}] 表格行完整性："
        f"{'没有任何一行被切断' if not broken_rows else f'{len(broken_rows)} 行被切断！'}",
        "",
        "被跳过的文件类型（load_document 不认，返回空列表）：",
        "  " + ", ".join(
            f"{s}×{n}" for s, n in sorted(by_suffix.items())
            if s not in {".md", ".markdown"}
        ),
    ]

    for cid, line in broken_rows[:5]:
        lines.append(f"  !! 残缺行 {cid}: {line[:60]}")

    report = "\n".join(lines) + "\n"
    REPORT.write_text(report, encoding="utf-8")

    # 控制台只打 ASCII 摘要，避免 Windows 终端编码把中文糊掉；完整报告看文件
    print(report.encode("ascii", "replace").decode("ascii"))
    print(f"完整报告（UTF-8）：{REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
