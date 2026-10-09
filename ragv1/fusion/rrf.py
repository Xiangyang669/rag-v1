"""RRF（Reciprocal Rank Fusion，倒数排名融合）。

    score(chunk) = Σ_paths 1 / (k + rank_in_path)

只用**排名**，不看分数——因此免疫三路的量纲差异（向量余弦、FTS5 bm25 负值、
图谱的命中数）。代价是丢掉绝对分数：看不出「第 1 名比第 2 名强多少」，也
看不出某个块是不是「压倒性匹配」。这个代价是有意接受的取舍。

融合输出**即最终排序**——V1 没有独立重排模块。
"""

from ragv1.config import RRF_K
from ragv1.types import FusedHit, Hit


def rrf_fuse(
    ranked: dict[str, list[Hit]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[FusedHit]:
    """把多路排序结果融合成一份最终排序。

    - 同一块跨路出现会被合并，`sources` 记下全部命中路径（排序后）
    - 打破平局用 chunk_id 升序 —— 保证同一输入结果完全可复现

    V2 加权：`weights` 给每路一个权重，贡献为 `w / (k + rank)`。
    - `weights is None`（或空 dict）→ 每路权重 1.0，结果与 V1 逐字节一致
    - 出现在 `ranked` 但不在 `weights` 里的路径 → 权重按 1.0
    - 权重 ≤ 0 的路径 → 视为不参与融合（跳过，而非贡献负分）
    - `weights` 里的未知路径名 → 静默忽略
    """
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    contributions: dict[str, dict[str, float]] = {}

    for path in sorted(ranked):
        weight = 1.0 if weights is None else weights.get(path, 1.0)
        if weight <= 0:
            continue
        for hit in ranked[path]:
            contribution = weight / (k + hit.rank)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + contribution
            sources.setdefault(hit.chunk_id, set()).add(path)
            contributions.setdefault(hit.chunk_id, {})[path] = contribution

    if not scores:
        return []

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        FusedHit(
            chunk_id=chunk_id,
            rrf_score=score,
            sources=tuple(sorted(sources[chunk_id])),
            rank=i + 1,
            contributions=contributions.get(chunk_id, {}),
        )
        for i, (chunk_id, score) in enumerate(ordered)
    ]
