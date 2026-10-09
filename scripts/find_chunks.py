"""标注辅助脚本：按查询找出候选块，打印 chunk_id / doc_id / page / 前 80 字原文。

用途：手工撰写 V2 人工评估集时，需要把「答案块」写成 chunk_id。本脚本把
RRF 融合后的检索结果连同它在**回源表**里的身份信息一并打印，省得另开一处查 ID。

用法：

    py scripts/find_chunks.py "怎么配置模型"
    py scripts/find_chunks.py "查询" --k 20
    py scripts/find_chunks.py "查询" --index-dir .indexes/kb --k 10

每行一条、按融合排序：

    <chunk_id> · <doc_id> · p<page> · <前 80 字原文>

⚠️ 只读：不写索引、不打任何 API key / .env。默认指向**中文索引**
`.indexes/kb_zh`——评估集是针对中文语料标注的；标英文基准时用 --index-dir 覆盖。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragv1 import config  # noqa: E402
from ragv1.api.server import build_retriever  # noqa: E402
from ragv1.fusion.rrf import rrf_fuse  # noqa: E402
from ragv1.ingest.build import FTS_DB  # noqa: E402
from ragv1.retrieve import ALL_PATHS  # noqa: E402
from ragv1.store.fts_store import FtsStore  # noqa: E402

# 打印进最后一列的原文上限（字符）。超出只用于**辨识**，不用于评估。
SNIPPET_CHARS = 80


def render_row(chunk_id: str, meta: dict | None, text: str | None) -> str:
    """把一条命中渲染成单行：`chunk_id · doc_id · p<page> · 前 80 字原文`。

    纯函数（无 I/O、无全局状态），因此单测可离线、确定性地直接调它。

    `meta` / `text` 为 None 时不崩、字段退化成 `?`——命中可能来自某条通路却
    不在回源表里（或回源表缺失该块），标注时看到一个带问号的行，好过看崩溃堆栈。
    """
    meta = meta or {}
    doc_id = meta.get("doc_id") or "?"
    page = meta.get("page")
    page_str = "?" if page is None else str(page)
    # 换行会把一条命中拆成多行、打乱逐行浏览；先压平所有空白成单行
    snippet = " ".join((text or "").split())
    if len(snippet) > SNIPPET_CHARS:
        snippet = snippet[:SNIPPET_CHARS] + "…"
    return f"{chunk_id} · {doc_id} · p{page_str} · {snippet}"


def main() -> int:
    # Windows 控制台默认 GBK，中文原文直接 print 会抛 UnicodeEncodeError。
    # 放在 argparse 之前——否则 `--help` 的中文说明先被打成乱码。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(
        description="按查询找候选块，打印 chunk_id / doc_id / page / 前 80 字（标注用）"
    )
    ap.add_argument("query", help="查询语句")
    ap.add_argument(
        "--index-dir",
        default=str(config.ZH_INDEX_DIR),
        help="索引目录（默认中文索引 .indexes/kb_zh）",
    )
    ap.add_argument("--k", type=int, default=10, help="返回前 k 条（默认 10）")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    # 索引不存在时 build_retriever 内部会 SystemExit 并提示先 build_index
    retriever = build_retriever(idx)
    # 回源表单独开一份：融合层不认识 store，原文 / 元数据只在这里查得到
    fts = FtsStore(idx / FTS_DB)

    fused = rrf_fuse(retriever.retrieve(args.query, args.k, ALL_PATHS))
    print(f"# 查询：{args.query}    索引：{idx}    融合命中：{len(fused)}")
    for fused_hit in fused[: args.k]:
        cid = fused_hit.chunk_id
        print(render_row(cid, fts.meta_of(cid), fts.text_of(cid)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
