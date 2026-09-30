"""RRF 倒数排名融合（硬约束三）。

用 RRF 而不是加权求和：三路分数量纲互不相同（向量余弦、FTS5 bm25 负值、
图谱的命中数），直接加权等于拿苹果加橘子。RRF 只用**排名**，天然免疫。

融合输出**即最终排序**——V1 没有独立重排模块。
"""

import pytest

from ragv1.types import Hit
from ragv1.fusion.rrf import rrf_fuse


def test_single_path_orders_by_rank():
    out = rrf_fuse({"vector": [Hit("a", 1, 0.9, "vector"), Hit("b", 2, 0.8, "vector")]}, k=60)
    assert [f.chunk_id for f in out] == ["a", "b"]
    assert out[0].sources == ("vector",)


def test_duplicate_chunk_merged_with_all_sources():
    out = rrf_fuse(
        {
            "vector": [Hit("a", 1, 0.9, "vector")],
            "graph": [Hit("a", 1, 1.0, "graph")],
        },
        k=60,
    )
    assert len(out) == 1
    assert set(out[0].sources) == {"vector", "graph"}


def test_scores_match_reciprocal_rank_formula():
    out = rrf_fuse(
        {
            "vector": [Hit("a", 1, 0.9, "vector")],
            "graph": [Hit("a", 1, 1.0, "graph")],
        },
        k=60,
    )
    assert out[0].rrf_score == pytest.approx(2 * (1 / 61))


def test_empty_input_returns_empty_list():
    assert rrf_fuse({}, k=60) == []
    assert rrf_fuse({"vector": []}, k=60) == []


def test_ties_are_deterministic():
    lists = {"vector": [Hit("a", 1, 0.9, "vector"), Hit("b", 2, 0.9, "vector")]}
    assert [f.chunk_id for f in rrf_fuse(lists, 60)] == [
        f.chunk_id for f in rrf_fuse(lists, 60)
    ]


def test_rank_field_is_filled_from_one():
    out = rrf_fuse({"vector": [Hit("a", 1, 0.9, "vector"), Hit("b", 2, 0.8, "vector")]}, k=60)
    assert [f.rank for f in out] == [1, 2]
