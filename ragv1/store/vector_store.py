"""向量索引（Chroma）。

Chroma 不负责算 embedding，所以向量由我们算好再传进去（见 embedding.py）。
embed_fn 可注入：测试传入确定性假向量，避免单测依赖网络与付费 API。

元数据随向量一起写入 Chroma 的 metadatas。⚠️ Chroma 只接受
str / int / float / bool 四种标量——元组与 None 都会报错，所以 bbox 与
degrade 序列化成 JSON 字符串、值为 None 的键直接省略（见 chunk_metadata）。
"""

import json
from pathlib import Path
from typing import Callable

import chromadb

from ragv1.types import Chunk, Hit

COLLECTION = "chunks"


def chunk_metadata(c: Chunk) -> dict:
    """把块的来源元数据转成 Chroma 能接受的形状。

    值为 None 的键**省略**而不是写成空串——空串会被当成「有值且为空」，
    与「没有这个字段」是两码事。
    """
    meta: dict = {
        "doc_id": c.doc_id,
        "kind": c.kind,
        "order": c.order,
        "part": c.part,
        "table_structured": bool(c.table_structured),
        "degrade": json.dumps(list(c.degrade)),
    }
    if c.page is not None:
        meta["page"] = c.page
    if c.bbox is not None:
        meta["bbox"] = json.dumps(list(c.bbox))
    if c.image_ref is not None:
        meta["image_ref"] = c.image_ref
    if c.image_path is not None:
        meta["image_path"] = c.image_path
    return meta


class VectorStore:
    def __init__(
        self,
        path: str | Path,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        if embed_fn is None:
            from ragv1.embedding import default_embed_fn

            embed_fn = default_embed_fn()
        self._embed = embed_fn

        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[self._embed(c.text) for c in chunks],
            metadatas=[chunk_metadata(c) for c in chunks],
        )

    def search(self, query: str, k: int) -> list[Hit]:
        if not query.strip():
            return []
        res = self._col.query(query_embeddings=[self._embed(query)], n_results=k)
        ids = res["ids"][0] if res["ids"] else []
        # cosine 空间下 distance = 1 - 相似度，转回相似度作为分数（越大越相关）
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        return [
            Hit(chunk_id=cid, rank=i + 1, score=1.0 - dist, path="vector")
            for i, (cid, dist) in enumerate(zip(ids, dists))
        ]
