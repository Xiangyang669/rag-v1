"""静态融合权重网格搜索（任务 8）。离线、确定性：不联网、不用索引、不读 key。"""

import importlib.util
from pathlib import Path

from ragv1.evaluation.qaset import EvalItem
from ragv1.types import Hit


def _load():
    spec = importlib.util.spec_from_file_location(
        "tune_weights", Path(__file__).resolve().parents[1] / "scripts" / "tune_weights.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_is_deterministic_and_descending():
    mod = _load()
    fake = [({"vector": 0.5}, {"recall@5": 0.4}), ({"vector": 0.9}, {"recall@5": 0.8})]
    out = mod.rank_results(fake)
    assert out[0][0] == {"vector": 0.9}


def test_grid_enumerates_all_combinations():
    mod = _load()
    combos = list(mod.weight_combos((0.3, 0.4), (0.5,)))
    assert len(combos) == 2
    assert all(abs(sum(c.values()) - 1.0) < 1e-9 for c in combos)
    assert all(c["graph"] >= 0.0 for c in combos)


def test_weight_combos_skips_negative_graph():
    mod = _load()
    # 0.9 + 0.9 > 1 → graph < 0 → 整组跳过（不允许负权重）
    assert list(mod.weight_combos((0.9,), (0.9,))) == []


def test_weight_combos_is_deterministic():
    mod = _load()
    a = mod.weight_combos((0.0, 0.5), (0.0, 0.5))
    b = mod.weight_combos((0.0, 0.5), (0.0, 0.5))
    assert a == b and len(a) == 4  # (0,0) (0,.5) (.5,0) (.5,.5) 全部 g≥0


def test_rank_results_tie_break_is_order_independent():
    mod = _load()
    rows = [({"vector": 0.5}, {"recall@5": 0.4}), ({"vector": 0.6}, {"recall@5": 0.4})]
    assert mod.rank_results(rows) == mod.rank_results(list(reversed(rows)))


def test_retrieval_items_excludes_unanswerable():
    mod = _load()
    items = (
        EvalItem("q1", ("a",), "fact"),
        EvalItem("q2", (), "unanswerable"),
    )
    kept = mod.retrieval_items(items)
    assert [it.question for it in kept] == ["q1"]


def test_score_weights_counts_hits_in_answers():
    mod = _load()
    items = (EvalItem("q", ("a",), "fact"),)
    cache = {"q": {"vector": [Hit("a", 1, 1.0, "vector")]}}
    metrics = mod.score_weights(cache, items, {"vector": 1.0}, (5, 10))
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr"] == 1.0


def test_score_weights_miss_is_zero():
    mod = _load()
    items = (EvalItem("q", ("a",), "fact"),)
    cache = {"q": {"vector": [Hit("x", 1, 1.0, "vector")]}}
    metrics = mod.score_weights(cache, items, {"vector": 1.0}, (5, 10))
    assert metrics["recall@5"] == 0.0 and metrics["mrr"] == 0.0


def test_best_all_path_skips_degenerate_corner():
    mod = _load()
    ranked = [
        ({"vector": 1.0, "fulltext": 0.0, "graph": 0.0}, {"recall@5": 0.8}),
        ({"vector": 0.8, "fulltext": 0.1, "graph": 0.1}, {"recall@5": 0.6}),
    ]
    assert mod.best_all_path(ranked)[0] == {"vector": 0.8, "fulltext": 0.1, "graph": 0.1}


class _CountingRetriever:
    """记录 retrieve 被调了几次——用来钉住「每条题只检索一次」。"""

    def __init__(self, ranked):
        self._ranked = ranked
        self.calls = 0

    def retrieve(self, query, k, paths=None):
        self.calls += 1
        return self._ranked[query]


def test_grid_retrieves_once_per_item_not_per_combo():
    mod = _load()
    items = (
        EvalItem("q1", ("a",), "fact"),
        EvalItem("q2", ("b",), "fact"),
    )
    ranked = {
        "q1": {"vector": [Hit("a", 1, 1.0, "vector")]},
        "q2": {"vector": [Hit("b", 1, 1.0, "vector")]},
    }
    retriever = _CountingRetriever(ranked)
    combos = mod.weight_combos((0.0, 0.5), (0.0, 0.5))  # 4 个组合
    out = mod.grid(retriever, items, (5, 10), combos)
    assert retriever.calls == 2  # 每条题一次，与组合数无关
    assert len(out) == 4
