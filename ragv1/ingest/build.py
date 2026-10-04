"""语料入库：目录 → 一套块 → 写进各路索引。

**硬约束一（三路共享前置）**：这里只 load + chunk 一次，得到唯一的
`all_chunks` 列表，再原样交给向量路与全文路。块边界与 chunk_id 因此
天然一致，融合时不会对不上。

数据流已从「parse → chunk」换成「load_document → chunk_elements」：
前者只认识 Markdown 正文，后者经 Element 中间表示，把表格作为独立元素
拆出来（并给每个块附上来源元数据）。

文件类型的分派交给 load_document 按扩展名处理——目录扫描因此放宽到所有
文件，未知类型返回空列表而不是报错。图谱路同样消费这个列表。
"""

from pathlib import Path
from typing import Callable

from ragv1.config import MAX_CHARS
from ragv1.ingest.elements import chunk_elements
from ragv1.ingest.loader import load_document
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
    """把 corpus_dir 下的文档入库，返回写入的块数。

    embed_fn 可注入：测试传确定性假向量，生产传 None 走真实 API。
    """
    corpus_dir = Path(corpus_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    # rglob("*") 会连目录一起命中，先过滤掉——否则名字里带 .md 的目录会在
    # read_text 时炸掉整次入库
    for path in sorted(p for p in corpus_dir.rglob("*") if p.is_file()):
        doc_id = path.relative_to(corpus_dir).as_posix()
        all_chunks.extend(
            chunk_elements(load_document(path, doc_id), doc_id, MAX_CHARS)
        )

    if not all_chunks:
        return 0

    # 同一个 all_chunks 交给两条路 —— 这是块 ID 对齐的来源
    VectorStore(index_dir / VECTOR_DIR, embed_fn=embed_fn).add(all_chunks)
    FtsStore(index_dir / FTS_DB).add(all_chunks)

    return len(all_chunks)
