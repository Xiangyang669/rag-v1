"""语料入库：目录 → 一套块 → 写进各路索引。

**硬约束一（三路共享前置）**：这里只 parse + chunk 一次，得到唯一的
`all_chunks` 列表，再原样交给向量路与全文路。块边界与 chunk_id 因此
天然一致，融合时不会对不上。

图谱路在 Task 3 接入，同样必须消费这个列表。
"""

from pathlib import Path
from typing import Callable

from ragv1.config import MAX_CHARS
from ragv1.ingest.chunker import chunk_document
from ragv1.ingest.parser import parse_document
from ragv1.store.fts_store import FtsStore
from ragv1.store.vector_store import VectorStore
from ragv1.types import Chunk

# 索引文件名
VECTOR_DIR = "vec"
FTS_DB = "kb.db"


def build_corpus(
    corpus_dir: str | Path,
    index_dir: str | Path,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> int:
    """把 corpus_dir 下所有 .md 入库，返回写入的块数。

    embed_fn 可注入：测试传确定性假向量，生产传 None 走真实 API。
    """
    corpus_dir = Path(corpus_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    for md in sorted(corpus_dir.rglob("*.md")):
        doc_id = md.relative_to(corpus_dir).as_posix()
        doc = parse_document(md.read_text(encoding="utf-8"), doc_id=doc_id)
        all_chunks.extend(chunk_document(doc, max_chars=MAX_CHARS))

    if not all_chunks:
        return 0

    # 同一个 all_chunks 交给两条路 —— 这是块 ID 对齐的来源
    VectorStore(index_dir / VECTOR_DIR, embed_fn=embed_fn).add(all_chunks)
    FtsStore(index_dir / FTS_DB).add(all_chunks)

    return len(all_chunks)
