"""FastAPI 检索端点。

V1 **不加鉴权、不做生产部署**（spec 的 Out-of-scope），只在本地跑得起来。

⚠️ `create_app` 接的是 **retriever**，不是预建好的图。原因：三路开关是在
**构建期**决定节点存不存在的（T9 —— 未启用的通路根本不在图里）。若接一个
预建图，请求里的 `paths` 就只能靠事后过滤结果来"实现"，而那会错误地丢掉
被「已启用路 + 已禁用路」同时命中的块。

**来源元数据**由注入的 `meta_lookup` 提供（生产传 `FtsStore.meta_of`）。
为什么不在融合层带出来：`rrf_fuse` 的契约是「只做排名、不认识 store」，
把存储依赖塞进去会毁掉它的可单测性。查表因此发生在 API 边界。

生产用法（需先 build_corpus 建好索引）：

    store = FtsStore(INDEX_DIR / "kb.db")
    app = create_app(retriever, meta_lookup=store.meta_of)
"""

from typing import Callable

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

    # ── 来源元数据 ──
    # 字段与 FtsStore.meta_of 的返回键一一对应，可直接展开注入。
    # 全部可选：没注入 meta_lookup 时保持 None，并在响应里被省略。
    doc_id: str | None = None
    kind: str | None = None
    page: int | None = None
    order: int | None = None
    part: int | None = None
    bbox: list[float] | None = None
    image_ref: str | None = None
    image_path: str | None = None
    table_structured: bool | None = None
    degrade: list[str] | None = None


class SearchResponse(BaseModel):
    results: list[ResultItem]
    # 被跳过的路径 → 原因。少了一路必须如实告知，不能只给一个变小的结果集。
    degraded: dict[str, str]


def create_app(
    retriever,
    meta_lookup: Callable[[str], dict | None] | None = None,
) -> FastAPI:
    app = FastAPI(title="rag-v1", description="三层检索 RAG（向量 / 全文 / 图谱）")

    # 按 paths 组合缓存编译好的图，避免每个请求都重新 compile
    graphs: dict[frozenset[str], object] = {}

    def graph_for(paths: frozenset[str]):
        if paths not in graphs:
            graphs[paths] = build_graph(retriever, paths)
        return graphs[paths]

    # exclude_none：没查到的元数据字段应当**整条省略**，而不是输出 null。
    # 「没有这个字段」和「有这个字段但值为空」对调用方是两回事。
    @app.post(
        "/search", response_model=SearchResponse, response_model_exclude_none=True
    )
    def search(req: SearchRequest) -> SearchResponse:
        if not req.query.strip():
            raise HTTPException(status_code=422, detail="query 不能为空")

        paths = ALL_PATHS if req.paths is None else (frozenset(req.paths) & ALL_PATHS)
        fused, degraded = run_query_meta(graph_for(paths), req.query, req.k)

        results = []
        for h in fused:
            # 查不到（None）时用空字典展开，字段保持默认的 None
            meta = (meta_lookup(h.chunk_id) if meta_lookup else None) or {}
            results.append(
                ResultItem(
                    chunk_id=h.chunk_id,
                    score=h.rrf_score,
                    sources=list(h.sources),
                    rank=h.rank,
                    **meta,
                )
            )

        return SearchResponse(results=results, degraded=degraded)

    return app
