"""查询自适应权重路由（任务 6）。

纯规则、零 LLM、零网络、确定性：按查询的**词面特征**与**全文路分数形态**，
为每条召回路产出融合权重，喂给 `rrf_fuse(..., weights=...)`（任务 5）。

为什么需要它：V1 的等权 RRF 实测**不如最强单路**（RRF 等权稀释——弱路把强路
的优势拉平了）。业界成熟做法是加权；本模块回答「权重从哪来」。

⚠️ 全文路 bm25 是**负值、越小越相关**（见 `ragv1/store/fts_store.py` 文件头）。
因此「top-1 明显强于其余」在负值下 = `median(scores) - top1` 是一个大的**正数**。
方向读反不会让任何指标（除测试外）报错——只会悄悄把权重调反、让融合变差。
"""

import re
import statistics

from ragv1.config import (
    FULLTEXT_DOMINANCE_GAP,
    LEXICAL_FULLTEXT_BOOST,
    LEXICAL_VECTOR_DAMP,
    REWRITE_WEIGHT_DISCOUNT,
    REWRITTEN_SUFFIX,
    STATIC_FUSION_WEIGHTS,
)
from ragv1.types import Hit

# ── 词面信号 ────────────────────────────────────────────────
# 代码标识符类查询里，字面 token 的精确匹配（FTS5）通常比语义向量更准。
# 四类：反引号包裹 / 含下划线 / 驼峰 / 全大写缩写（≥2 字符）。
_BACKTICK_RE = re.compile(r"`[^`]+`")
_UNDERSCORE_RE = re.compile(r"[A-Za-z0-9]+_[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"[a-z][A-Z]")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")

# 「压倒性命中」至少要有这么多条全文命中才判——2 条时 median 只由两点决定，不稳。
_MIN_FULLTEXT_FOR_DOMINANCE = 3


def _has_lexical_signal(query: str) -> bool:
    """查询是否带强词面信号（代码标识符类）。"""
    return bool(
        _BACKTICK_RE.search(query)
        or _UNDERSCORE_RE.search(query)
        or _CAMEL_RE.search(query)
        or _ACRONYM_RE.search(query)
    )


def _fulltext_is_dominant(hits: list[Hit]) -> bool:
    """全文路 top-1 是否「压倒性」强于其余。

    bm25 负值、越小越相关 → top-1 是**最小**的分数。当它与其余的中位数拉开
    `FULLTEXT_DOMINANCE_GAP` 以上时，说明有一个块明显匹配（这种强命中在等权
    RRF 里会被弱路稀释掉），值得给全文路加权。

    判据写作 `median(scores) - top1 > gap`：负值下 `median` 比 `top1` 更接近 0，
    相减得到**正**的大数；将来若换成正分打器，方向不对时自然不触发（不会反向调）。
    """
    if len(hits) < _MIN_FULLTEXT_FOR_DOMINANCE:
        return False
    ordered = sorted(hits, key=lambda h: h.rank)  # top-1 = rank 最小者
    scores = [h.score for h in ordered]
    return statistics.median(scores) - scores[0] > FULLTEXT_DOMINANCE_GAP


def route_weights(
    query: str,
    ranked: dict[str, list[Hit]],
    static: dict[str, float] | None = None,
) -> dict[str, float]:
    """按查询特征与各路表现，决定每一路的融合权重（确定性、零 LLM、零网络）。

    - 从 `static`（缺省 `config.STATIC_FUSION_WEIGHTS`）的副本出发；
    - **词面信号**（反引号 / 下划线 / 驼峰 / 全大写缩写）→ 全文路 ×= boost、
      向量路 ×= damp；
    - **全文路压倒性命中**（`ranked["fulltext"]` ≥ 3 条且 `median - top1 > gap`）
      → 同上调整；
    - **改写变体**：`ranked` 里带 `REWRITTEN_SUFFIX` 的键 → 权重 =
      同路径权重 × discount；不在 `ranked` 里的变体键不产出。

    `ranked` 里出现但 `static` 未覆盖的路 → 权重 1.0（与 `rrf_fuse` 缺省一致），
    保证结果覆盖 `ranked` 里的每一条路。

    调整只作用于**已存在**的键（`static` ∪ `ranked`）——不凭空发明路径名。
    """
    weights = dict(STATIC_FUSION_WEIGHTS if static is None else static)
    for path in ranked:  # 覆盖 ranked 的每条路（含变体）；非变体缺省按 1.0
        weights.setdefault(path, 1.0)

    lexical = _has_lexical_signal(query)
    dominant = _fulltext_is_dominant(ranked.get("fulltext", []))
    if lexical or dominant:
        if "fulltext" in weights:
            weights["fulltext"] *= LEXICAL_FULLTEXT_BOOST
        if "vector" in weights:
            weights["vector"] *= LEXICAL_VECTOR_DAMP

    # 改写变体：只在它确实出现在 ranked 时产出，继承同路径（可能已调整）的权重。
    for path in ranked:
        if path.endswith(REWRITTEN_SUFFIX):
            stem = path[: -len(REWRITTEN_SUFFIX)]
            weights[path] = weights.get(stem, 1.0) * REWRITE_WEIGHT_DISCOUNT

    return weights
