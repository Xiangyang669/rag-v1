"""生产装配：把落盘的三路索引组装成一个可服务的 FastAPI app。

**为什么需要这个文件**：`api/main.py` 的 `create_app` 只接受**注入好的**
retriever —— 测试里够用，但**生产上从没有一处把 store → retriever → app
接起来过**。缺了这一步，服务根本起不来（"部署"这块空白的一半就是它）。

索引布局（由 `ingest/build.py` 写入、`scripts/build_index.py` 触发）：

    <index_dir>/vec/        Chroma 持久化目录
    <index_dir>/kb.db       SQLite FTS5 —— 同时是唯一的**回源表**
    <index_dir>/graph.db    SQLite 图谱（nodes / edges / entity_chunks）

⚠️ 图谱路要一份 chunk_id 集合做 membership 判定（`GraphRetriever.retrieve`
只做 `chunk_id in self._chunks`）。FtsStore 的 `chunks` 表是项目里唯一的
回源表，所以从它重建——**不另建一份持久化，避免两处真相**。
`heading_path` 没有落盘，而图谱路用不到它，置空即可。
"""

import os
from pathlib import Path

from ragv1.api.main import create_app
from ragv1.ingest.build import FTS_DB, GRAPH_DB, VECTOR_DIR
from ragv1.retrieve import Retriever
from ragv1.retrieve.fts_retriever import FtsRetriever
from ragv1.retrieve.graph_retriever import GraphRetriever
from ragv1.retrieve.vector_retriever import VectorRetriever
from ragv1.store.fts_store import FtsStore
from ragv1.store.graph_store import GraphStore
from ragv1.store.vector_store import VectorStore
from ragv1.types import Chunk

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / ".indexes" / "kb"


def index_dir() -> Path:
    """索引目录：`RAGV1_INDEX_DIR` 环境变量优先（容器里靠它挂载）。"""
    return Path(os.environ.get("RAGV1_INDEX_DIR", str(DEFAULT_INDEX_DIR)))


def _chunks_by_id(fts: FtsStore) -> dict[str, Chunk]:
    """从唯一的回源表重建 chunk 映射，供图谱路做 membership 判定。"""
    out: dict[str, Chunk] = {}
    for cid in fts.chunk_ids():
        meta = fts.meta_of(cid) or {}
        bbox = meta.get("bbox")
        out[cid] = Chunk(
            chunk_id=cid,
            doc_id=meta.get("doc_id", ""),
            heading_path=(),  # 未落盘；图谱路只做 membership，用不到
            text=fts.text_of(cid) or "",
            kind=meta.get("kind", "text"),
            page=meta.get("page"),
            order=meta.get("order", 0),
            part=meta.get("part", 0),
            bbox=tuple(bbox) if bbox else None,
            image_ref=meta.get("image_ref"),
            image_path=meta.get("image_path"),
            table_structured=meta.get("table_structured", True),
            degrade=tuple(meta.get("degrade") or ()),
        )
    return out


def build_app(idx: str | Path | None = None, embed_fn=None):
    """装配：索引目录 → 三路 store → Retriever → FastAPI app。

    `embed_fn` 可注入（与 `build_corpus` 同一约定）：生产传 None 走真实 API，
    测试传确定性假向量——否则这条装配路径**没法在离线环境被测试**。
    """
    idx = Path(idx) if idx is not None else index_dir()
    if not idx.is_dir():
        raise SystemExit(f"索引目录不存在：{idx}\n先跑 `py scripts/build_index.py`")

    fts = FtsStore(idx / FTS_DB)
    retriever = Retriever(
        VectorRetriever(VectorStore(idx / VECTOR_DIR, embed_fn=embed_fn)),
        FtsRetriever(fts),
        GraphRetriever(GraphStore(idx / GRAPH_DB), _chunks_by_id(fts)),
    )
    # meta_lookup 走 FtsStore —— 查表发生在 API 边界，融合层不认识 store
    return create_app(retriever, meta_lookup=fts.meta_of)


def main() -> None:
    import uvicorn

    uvicorn.run(
        build_app(),
        host=os.environ.get("RAGV1_HOST", "127.0.0.1"),
        port=int(os.environ.get("RAGV1_PORT", "8000")),
        log_level=os.environ.get("RAGV1_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
