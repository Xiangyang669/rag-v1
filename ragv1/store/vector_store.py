"""向量索引（Chroma）。

Chroma 不负责算 embedding，所以向量由我们算好再传进去（见 embedding.py）。
embed_fn 可注入：测试传入确定性假向量，避免单测依赖网络与付费 API。
"""

from pathlib import Path
from typing import Callable

import chromadb

from ragv1.types import Chunk, Hit

COLLECTION = "chunks"


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
