"""建**真实**索引：走 SiliconFlow 真 embedding（`bge-m3`），落盘到 `.indexes/kb`。

与 `scripts/verify_ingest.py` 的分工（**这个区别是重点**）：

- `verify_ingest` 传 `fake_embed`（确定性假向量）——它验的是**入库链路**
  （表格/图片切得对不对、降级码记没记），**不是检索质量**。
- 本脚本传 `embed_fn=None` → **真调 API**。索引里的向量是真的，**能真检索**，
  也才能拿来压测。

用法：

    py scripts/build_index.py                 # 全量 107 篇，不解析图片（快）
    py scripts/build_index.py --with-images   # 带图片双通道（取图 40-68s/张，很慢）

⚠️ 需要 `SILICONFLOW_API_KEY`（本目录 `.env` 或上层 `practice/.env`）。
⚠️ 索引写在 `.indexes/kb/`（已 gitignore），可重复执行：每次**先清空**该目录。
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragv1 import config  # noqa: E402
from ragv1.embedding import EMBED_MODEL, MAX_EMBED_BATCH, default_embed_fn  # noqa: E402
from ragv1.ingest.build import build_corpus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="建真实索引（真 embedding）")
    ap.add_argument("--corpus", default=str(config.CORPUS_DIR), help="语料目录")
    ap.add_argument(
        "--index-dir", default=str(ROOT / ".indexes" / "kb"), help="索引落盘目录"
    )
    ap.add_argument(
        "--with-images", action="store_true", help="真跑 OCR/多模态（取图很慢）"
    )
    args = ap.parse_args()

    corpus = Path(args.corpus)
    index_dir = Path(args.index_dir)
    if not corpus.is_dir():
        print(f"[错误] 语料目录不存在：{corpus}", file=sys.stderr)
        return 2

    ocr = vlm = None
    if args.with_images:
        from ragv1.ingest.ocr import build_ocr_engine
        from ragv1.ingest.vlm import build_vision_engine

        ocr, vlm = build_ocr_engine(), build_vision_engine()

    # 每次重建：保证同一份语料得到同一份索引，数字可复现
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)

    stats = {"calls": 0, "texts": 0}
    base = default_embed_fn()

    def embed(texts: list[str]) -> list[list[float]]:
        """按 MAX_EMBED_BATCH 自己分批调用 base —— 这样计的是**真实 API 往返数**。

        ⚠️ 不能只在最外层包一层：`default_embed_fn` 内部本来就会按 32 切片，
        外层看到的永远只是"1 次调用"，会把 65 次往返报成 1 次。
        """
        vectors: list[list[float]] = []
        for i in range(0, len(texts), MAX_EMBED_BATCH):
            batch = texts[i : i + MAX_EMBED_BATCH]
            stats["calls"] += 1
            stats["texts"] += len(batch)
            vectors.extend(base(batch))
        return vectors

    print(f"语料   : {corpus}")
    print(f"模型   : {EMBED_MODEL}（单次批量 ≤ {MAX_EMBED_BATCH}）")
    print(f"索引   : {index_dir}")
    print(f"图片   : {'开' if args.with_images else '关（按正文处理）'}")
    print("建索引中……")

    started = time.time()
    total = build_corpus(corpus, index_dir, embed_fn=embed, ocr=ocr, vlm=vlm)
    elapsed = time.time() - started

    size = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file())
    print()
    print(f"块数           : {total}")
    print(f"入库耗时       : {elapsed:.1f}s")
    print(f"embedding 调用 : {stats['calls']} 次（共 {stats['texts']} 条文本）")
    if stats["calls"]:
        print(f"单次往返       : {elapsed / stats['calls'] * 1000:.0f} ms")
    print(f"索引落盘       : {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
