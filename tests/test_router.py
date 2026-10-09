"""查询自适应权重路由（任务 6）。

纯规则、零 LLM、确定性：按查询词面特征与全文路分数形态，决定每一路的权重。
输出喂给 `rrf_fuse(..., weights=...)`（任务 5）。

⚠️ 全文路 bm25 是**负值，越小越相关**（见 `ragv1/store/fts_store.py` 文件头）。
「top-1 明显强于其余」在负值下对应 `median(scores) - top1` 是一个大的**正数**。
方向读反不会让测试以外的指标报错——只会悄悄把权重调反。
"""

from ragv1.fusion.router import route_weights
from ragv1.types import Hit


def _h(cid, rank, path, score):
    return Hit(chunk_id=cid, rank=rank, score=score, path=path)


BASE = {"vector": 0.50, "fulltext": 0.35, "graph": 0.15}


def test_static_baseline_when_no_signal():
    w = route_weights("数据库连接怎么配", {}, static=dict(BASE))
    assert w["vector"] == 0.50


def test_identifier_query_upweights_fulltext():
    w = route_weights("MAX_EMBED_BATCH 是多少", {}, static=dict(BASE))
    assert w["fulltext"] > BASE["fulltext"]


def test_backticked_term_upweights_fulltext():
    w = route_weights("`resolve_api_key` 怎么用", {}, static=dict(BASE))
    assert w["fulltext"] > BASE["fulltext"]


# ⚠️ Review Focus 1：bm25 是负值，越小越相关。方向读反会让权重朝反方向调。
def test_dominant_fulltext_hit_upweights_fulltext_bm25_is_negative():
    ranked = {"fulltext": [_h("a", 1, "fulltext", -9.0),
                           _h("b", 2, "fulltext", -3.0),
                           _h("c", 3, "fulltext", -3.0),
                           _h("d", 4, "fulltext", -3.0)]}
    w = route_weights("随便的问题", ranked, static=dict(BASE))
    assert w["fulltext"] > BASE["fulltext"]


def test_flat_fulltext_scores_do_not_upweight():
    ranked = {"fulltext": [_h("a", 1, "fulltext", -3.0), _h("b", 2, "fulltext", -3.1)]}
    w = route_weights("随便的问题", ranked, static=dict(BASE))
    assert w["fulltext"] == BASE["fulltext"]


def test_rewritten_variant_is_discounted():
    w = route_weights("追问", {"vector": [_h("a", 1, "vector", 0.9)]}, static=dict(BASE))
    assert "vector#rw" not in w        # 只有变体存在时才产出它


def test_rewritten_variant_discount_applied():
    ranked = {"vector#rw": [_h("a", 1, "vector#rw", 0.9)]}
    w = route_weights("追问", ranked, static=dict(BASE))
    assert w["vector#rw"] < w["vector"]


def test_is_deterministic():
    ranked = {"fulltext": [_h("a", 1, "fulltext", -9.0)]}
    assert route_weights("q", ranked, static=dict(BASE)) == route_weights("q", ranked, static=dict(BASE))


def test_result_covers_every_path_present_in_ranked():
    ranked = {"graph": [_h("g", 1, "graph", 1.0)]}
    assert "graph" in route_weights("q", ranked, static=dict(BASE))
