"""RRF（Reciprocal Rank Fusion，倒数排名融合）。

    score(chunk) = Σ_paths 1 / (k + rank_in_path)

只用**排名**，不看分数——因此免疫三路的量纲差异（向量余弦、FTS5 bm25 负值、
图谱的命中数）。代价是丢掉绝对分数：看不出「第 1 名比第 2 名强多少」，也
看不出某个块是不是「压倒性匹配」。这个代价是有意接受的取舍。

融合输出**即最终排序**——V1 没有独立重排模块。
"""

from ragv1.config import RRF_K
from ragv1.types import FusedHit, Hit


def rrf_fuse(ranked: dict[str, list[Hit]], k: int = RRF_K) -> list[FusedHit]:
    """把多路排序结果融合成一份最终排序。

    - 同一块跨路出现会被合并，`sources` 记下全部命中路径（排序后）
    - 打破平局用 chunk_id 升序 —— 保证同一输入结果完全可复现
    """
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}

    for path in sorted(ranked):
        for hit in ranked[path]:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
            sources.setdefault(hit.chunk_id, set()).add(path)

    if not scores:
        return []

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        FusedHit(
            chunk_id=chunk_id,
            rrf_score=score,
            sources=tuple(sorted(sources[chunk_id])),
            rank=i + 1,
        )
        for i, (chunk_id, score) in enumerate(ordered)
    ]
