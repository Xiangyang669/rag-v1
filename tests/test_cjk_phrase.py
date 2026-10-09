"""中文短语匹配：修 CJK 词边界 bug（M2.5 设计 §5）。

背景：`_contains_phrase` 用两侧补空格做词边界——对英文是对的（它正是防
`url` 匹配 `curling` 的机制），但**中文按字连写、没有词间空格**，于是连
「用户把标题一字不差说出来」都匹配不到：

    _normalize('(1)如何检查模型可用性问题') = '1 如何检查模型可用性问题'
    _contains_phrase('……是怎么实现的', '1 如何检查模型可用性问题') = False

修法（**不是放宽匹配提升命中率**，是修一个结构性失效）：含够长 CJK 段的
别名改走「连续 CJK 段子串」这条路径；其余（纯 ASCII、或 CJK 段都太短）
逐字节沿用既有的空格词边界。

三条边界：
- 编号前缀（`2 前端错误`）不含 CJK，被「取连续 CJK 段」天然跳过。
- `MIN_CJK_RUN` 挡住「介绍」「说明」这类短标题在全库泛匹配——宁可漏，不可错。
- ASCII 词边界行为逐字节不变——`url` 绝不匹配上 `curling`。
"""

import pytest

from ragv1.retrieve.graph_retriever import _contains_phrase, _cjk_runs
from ragv1 import config


def test_cjk_runs_extracts_contiguous_chinese():
    assert _cjk_runs("1 如何检查模型可用性问题") == ["如何检查模型可用性问题"]


def test_cjk_runs_skips_punctuation_and_digits():
    assert _cjk_runs("2 报错 模型响应为空 模型报错") == ["报错", "模型响应为空", "模型报错"]


# ⚠️ Review Focus 1：编号前缀必须被天然跳过
def test_heading_with_number_prefix_matches_by_its_chinese_part():
    assert _contains_phrase("前端错误怎么排查", "2 前端错误") is True


def test_verbatim_heading_in_longer_query_matches():
    """§1.2 的原样复现——这条在改动前是红的。"""
    assert _contains_phrase("如何检查模型可用性问题是怎么实现的", "1 如何检查模型可用性问题") is True


# ⚠️ Review Focus 5：短标题不得泛匹配
def test_short_chinese_run_does_not_match():
    assert _contains_phrase("介绍一下这个系统", "介绍") is False


def test_run_shorter_than_threshold_does_not_match():
    short = "三字词"
    assert len(short) < config.MIN_CJK_RUN
    assert _contains_phrase(f"关于{short}的说明", short) is False


# ⚠️ Review Focus 2：英文词边界必须逐字节不变
def test_ascii_word_boundary_is_unchanged():
    assert _contains_phrase("how does browser work", "browser") is True


def test_ascii_does_not_match_inside_a_word():
    assert _contains_phrase("curling is fun", "url") is False


def test_mixed_alias_uses_ascii_path_when_no_cjk():
    assert _contains_phrase("set appid here", "appid") is True


# 混合别名（CJK 段太短、够不着新路径）不得因分流而丢失既有的整串空格边界。
# 既有测试 test_multi_word_entity_matches 走的就是别名 "requests 模块"（CJK
# 段仅 2 字）——分流若把这类别名一律赶进 CJK 路径，会静默回归。
def test_short_cjk_run_in_mixed_alias_falls_back_to_word_boundary():
    assert _contains_phrase("requests 模块 是干什么的", "requests 模块") is True


def test_empty_needle_returns_false():
    assert _contains_phrase("任意", "") is False


def test_deterministic():
    args = ("如何检查模型可用性问题是怎么实现的", "1 如何检查模型可用性问题")
    assert _contains_phrase(*args) == _contains_phrase(*args)
