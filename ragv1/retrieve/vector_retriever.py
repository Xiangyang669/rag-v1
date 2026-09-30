"""向量召回通路。薄封装——检索策略在 store，编排在 supervisor。"""

from ragv1.store.vector_store import VectorStore
from ragv1.types import Hit


class VectorRetriever:
    def __init__(self, store: VectorStore):
        self._store = store

    def retrieve(self, query: str, k: int) -> list[Hit]:
        return self._store.search(query, k)
