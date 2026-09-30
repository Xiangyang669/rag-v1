"""图谱召回通路。

做法：从 query 里**宽松**匹配出实体（分词后扫别名表），取这些实体以及
它们**一跳邻居**所挂的块。邻居那一跳是图谱路的独占价值——用户没提到、
但结构上相关的块。

⚠️ 最大失效模式：口语化提问（「这个功能怎么设置」）抽不到任何实体。
此时**必须返回空列表**，不能抛异常——否则整条路会拖垮一次查询。
"""

from ragv1.store.graph_store import GraphStore
from ragv1.text.tokenize import tokenize_for_index
from ragv1.types import Chunk, Hit


class GraphRetriever:
    def __init__(self, store: GraphStore, chunks_by_id: dict[str, Chunk]):
        self._store = store
        self._chunks = chunks_by_id

    def _matched_entities(self, query: str) -> set[str]:
        aliases = self._store.aliases()
        tokens = tokenize_for_index(query).split()
        return {aliases[t] for t in tokens if t in aliases}

    def retrieve(self, query: str, k: int) -> list[Hit]:
        if not query.strip():
            return []

        matched = self._matched_entities(query)
        if not matched:
            return []

        # 命中实体 + 各自的一跳邻居
        entities: set[str] = set(matched)
        for entity in sorted(matched):
            entities.update(self._store.neighbors(entity))

        # 一个块被越多命中实体指向，排名越前
        counts: dict[str, int] = {}
        for entity in sorted(entities):
            for chunk_id in self._store.chunks_of(entity):
                if chunk_id in self._chunks:
                    counts[chunk_id] = counts.get(chunk_id, 0) + 1

        # 命中数降序、块 ID 升序（破平局）——保证同一 query 结果可复现
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            Hit(chunk_id=chunk_id, rank=i + 1, score=float(n), path="graph")
            for i, (chunk_id, n) in enumerate(ordered)
        ]
