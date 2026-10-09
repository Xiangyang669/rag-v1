"""LangGraph 编排层（Supervisor 模式）。

**定位**：三层检索的流程调度节点。三路从 START **并行扇出**，各自把结果
写进归一化的 `ranked`，再汇入融合节点。

⚠️ 编排层**只做调度**，不含任何检索逻辑——检索在 retrieve/ 里。
⚠️ 三条通路是**并行**的，不是串行兜底阶段。串行会让一次查询的延迟变成
三路之和；tests/test_supervisor.py 用计时钉住了这一点。
⚠️ **降级只在这一层兜**：单路异常时该路缺席、其余照常融合。但只捕获
检索节点的异常——关键路径的错误不能被吞掉。
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ragv1 import observability
from ragv1.fusion.router import route_weights
from ragv1.fusion.rrf import rrf_fuse
from ragv1.retrieve import ALL_PATHS, PATH_FULLTEXT, PATH_GRAPH, PATH_VECTOR
from ragv1.types import FusedHit, Hit

FUSE_NODE = "fuse"

# on_error 的取值：单路失败时跳过（默认）还是直接抛出
ON_ERROR_SKIP = "skip"
ON_ERROR_RAISE = "raise"

# 路径名 → Retriever 上的属性名（全文路叫 fts，不是 fulltext）
_ATTR_BY_PATH = {
    PATH_VECTOR: "vector",
    PATH_FULLTEXT: "fts",
    PATH_GRAPH: "graph",
}


def _merge_dict(left: dict | None, right: dict | None) -> dict:
    """并行分支的状态合并：各路写进去的键互不重叠，直接并起来即可。"""
    return {**(left or {}), **(right or {})}


class State(TypedDict):
    query: str
    k: int
    on_error: str
    # ranked / degraded 由三路并行扇出、各写自己的键 → 需要 _merge_dict 归约。
    ranked: Annotated[dict[str, list[Hit]], _merge_dict]
    degraded: Annotated[dict[str, str], _merge_dict]
    # weights 只由融合节点这一个写入者产出（与 fused 同理），无并行写入者
    # → 不带归约器；带了反而是在暗示不存在的并行写。
    weights: dict[str, float]
    fused: list[FusedHit]


def _retrieval_node(retriever, path: str, tracer=None):
    """一个只负责「调一路、把结果写进 ranked」的节点。"""

    def node(state: State) -> dict:
        try:
            # span 名用路径名 —— 一次查询在 Langfuse 上能看出三路各自的耗时
            with observability.span(tracer, path):
                hits = getattr(retriever, _ATTR_BY_PATH[path]).retrieve(
                    state["query"], state["k"]
                )
        except Exception as err:  # noqa: BLE001 —— 降级点：只兜检索节点
            if state.get("on_error", ON_ERROR_SKIP) == ON_ERROR_RAISE:
                raise
            return {
                "ranked": {path: []},
                "degraded": {path: f"{type(err).__name__}: {err}"},
            }
        return {"ranked": {path: hits}}

    return node


def _fuse_node(state: State) -> dict:
    """归一化 + 融合。这是三路汇合后的唯一出口。

    融合前先按查询特征与全文路分数形态定权（`router.route_weights`，纯规则、
    零 LLM、零网络），再把权重连同结果写回 state —— 权重是调参时唯一的观测点。
    只算权重不写回，事后就无法解释「这次为什么是这个排序」。
    """
    weights = route_weights(state["query"], state["ranked"])
    return {"fused": rrf_fuse(state["ranked"], weights=weights), "weights": weights}


def build_graph(retriever, paths, tracer=None):
    """按 paths 构建编排图。

    只给**启用的**路径建节点——未启用的通路根本不在图里，从结构上保证
    它不会被调用（而不是靠节点内部 if 判断）。

    tracer 缺省时按环境自动构造；没有 Langfuse key 就是 no-op。
    """
    active = sorted(set(paths) & ALL_PATHS)
    if tracer is None:
        tracer = observability.tracer()

    builder = StateGraph(State)
    for path in active:
        builder.add_node(path, _retrieval_node(retriever, path, tracer))
    builder.add_node(FUSE_NODE, _fuse_node)

    if active:
        for path in active:
            builder.add_edge(START, path)  # 并行扇出
            builder.add_edge(path, FUSE_NODE)  # 汇入融合
    else:
        builder.add_edge(START, FUSE_NODE)

    builder.add_edge(FUSE_NODE, END)
    return builder.compile()


def run_query_meta(graph, query: str, k: int, on_error: str = ON_ERROR_SKIP):
    """跑一次查询，同时把降级情况交回给调用方。

    返回 `(融合结果, {被跳过的路径: 错误信息})`。API 层要如实告知使用者
    「这次少了一路」——只给一个变小的结果集等于隐瞒。
    """
    state = graph.invoke(
        {"query": query, "k": k, "on_error": on_error, "ranked": {}, "degraded": {}}
    )
    return state.get("fused", []), dict(state.get("degraded") or {})


def run_query(graph, query: str, k: int, on_error: str = ON_ERROR_SKIP) -> list[FusedHit]:
    """跑一次查询，返回融合后的最终排序。"""
    fused, _ = run_query_meta(graph, query, k, on_error=on_error)
    return fused
