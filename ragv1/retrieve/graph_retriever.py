"""图谱召回通路。

做法：从 query 里**宽松**匹配出实体，取这些实体以及它们**一跳邻居**所挂
的块。邻居那一跳是图谱路的独占价值——用户没提到、但结构上相关的块。

⚠️ 最大失效模式：口语化提问（「这个功能怎么设置」）抽不到任何实体。
此时**必须返回空列表**，不能抛异常——否则整条路会拖垮一次查询。

⚠️ 匹配必须是**宽松**的，不是「分词后整串等于别名」：section 的别名是
整个标题（如 `Admin Service`），分词后得到 `Admin`、`Service`——没有
任何单个 token 等于整串，那样多词实体就永远匹配不上。
"""

import re

from ragv1.store.graph_store import GraphStore
from ragv1.types import Chunk, Hit

# 归一化时保留：词字符（Python 的 \w 含 CJK）、点、连字符、下划线。
# 其余（括号、问号、逗号、斜杠…）一律化为空格，这样
# `requests.get(url)` 里的括号不会挡住 `requests.get` 的匹配。
_DROP = re.compile(r"[^\w.\-]+")
_SPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _SPACE.sub(" ", _DROP.sub(" ", text.lower())).strip()


def _contains_phrase(haystack: str, needle: str) -> bool:
    """needle 是否作为**独立词序列**出现在 haystack 里。

    两侧补空格做词边界，避免 `url` 匹配上 `curling` 这种词内出现。
    """
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


class GraphRetriever:
    def __init__(self, store: GraphStore, chunks_by_id: dict[str, Chunk]):
        self._store = store
        self._chunks = chunks_by_id

    def _matched_entities(self, query: str) -> set[str]:
        """宽松匹配 query 里出现的实体，返回它们的 canonical。"""
        normalized = _normalize(query)
        if not normalized:
            return set()

        matched: set[str] = set()
        for alias, canonical in self._store.aliases().items():
            if _contains_phrase(normalized, _normalize(alias)):
                matched.add(canonical)
        return matched

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
