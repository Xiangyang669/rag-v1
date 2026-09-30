"""三路召回的入口。

`Retriever` 按 `paths` 有选择地调用三条通路，只把**实际调用**的路径放进
返回字典——这是消融对比（「关掉某一路后指标掉多少」）的基础。

⚠️ 三个子检索器以公开属性暴露（`vector` / `fts` / `graph`）。T9 的
Supervisor 需要按路径**独立**调用它们才能做真正的并行扇出；若只能经
`retrieve()`（内部是顺序循环），扇出会退化成串行。
"""

from ragv1.types import Hit

# 三条通路的路径名。注意 "fulltext" 对应 fts 属性。
PATH_VECTOR = "vector"
PATH_FULLTEXT = "fulltext"
PATH_GRAPH = "graph"

ALL_PATHS: frozenset[str] = frozenset({PATH_VECTOR, PATH_FULLTEXT, PATH_GRAPH})


class Retriever:
    def __init__(self, vector, fts, graph):
        self.vector = vector
        self.fts = fts
        self.graph = graph
        self._by_path = {
            PATH_VECTOR: vector,
            PATH_FULLTEXT: fts,
            PATH_GRAPH: graph,
        }

    def retrieve(
        self, query: str, k: int, paths: set[str] | frozenset[str] = ALL_PATHS
    ) -> dict[str, list[Hit]]:
        """按 paths 调用对应通路，返回 {路径名: 命中列表}。

        未启用的路径**不会被调用**（有测试用计数钉住）。
        """
        enabled = sorted(ALL_PATHS & set(paths))
        return {path: self._by_path[path].retrieve(query, k) for path in enabled}
