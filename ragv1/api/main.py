"""FastAPI 检索端点。

V1 **不加鉴权、不做生产部署**（spec 的 Out-of-scope），只在本地跑得起来。

⚠️ `create_app` 接的是 **retriever**，不是预建好的图。原因：三路开关是在
**构建期**决定节点存不存在的（T9 —— 未启用的通路根本不在图里）。若接一个
预建图，请求里的 `paths` 就只能靠事后过滤结果来"实现"，而那会错误地丢掉
被「已启用路 + 已禁用路」同时命中的块。所以按请求（带缓存地）建图。
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ragv1.orchestration.supervisor import build_graph, run_query_meta
from ragv1.retrieve import ALL_PATHS

DEFAULT_K = 10


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=DEFAULT_K, ge=1, le=100)
    # None 表示用全量；[] 表示显式不要任何一路
    paths: list[str] | None = None


class ResultItem(BaseModel):
    chunk_id: str
    score: float
    sources: list[str]
    rank: int


class SearchResponse(BaseModel):
    results: list[ResultItem]
    # 被跳过的路径 → 原因。少了一路必须如实告知，不能只给一个变小的结果集。
    degraded: dict[str, str]


def create_app(retriever) -> FastAPI:
    app = FastAPI(title="rag-v1", description="三层检索 RAG（向量 / 全文 / 图谱）")

    # 按 paths 组合缓存编译好的图，避免每个请求都重新 compile
    graphs: dict[frozenset[str], object] = {}

    def graph_for(paths: frozenset[str]):
        if paths not in graphs:
            graphs[paths] = build_graph(retriever, paths)
        return graphs[paths]

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        if not req.query.strip():
            raise HTTPException(status_code=422, detail="query 不能为空")

        paths = ALL_PATHS if req.paths is None else (frozenset(req.paths) & ALL_PATHS)
        fused, degraded = run_query_meta(graph_for(paths), req.query, req.k)

        return SearchResponse(
            results=[
                ResultItem(
                    chunk_id=h.chunk_id,
                    score=h.rrf_score,
                    sources=list(h.sources),
                    rank=h.rank,
                )
                for h in fused
            ],
            degraded=degraded,
        )

    return app
