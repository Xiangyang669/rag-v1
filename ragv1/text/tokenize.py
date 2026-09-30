"""中文分词。

FTS5 的 `unicode61` 分词器把**连续 CJK 串当成一个整 token**，于是整串能
匹配、任何子串或分词后的查询都**静默返回 0 条**——不报错，只是召回为空。

解法：入库前先分词、查询侧走同一个函数。两边口径一致才可能命中。

只切 CJK 段；ASCII 段原样保留（`requests.get` 这种标识符不能被切碎）。
"""

import re

import jieba

# 连续 CJK 段
_CJK_RUN = re.compile(r"[一-鿿]+")
# 以 CJK 段为界做切分，保留分隔符
_CJK_BOUNDARY = re.compile(r"([一-鿿]+)")


def tokenize_for_index(text: str) -> str:
    """转成空格分隔的 token 串，供 FTS5 入库与查询共用。"""
    tokens: list[str] = []
    for part in _CJK_BOUNDARY.split(text):
        if not part:
            continue
        if _CJK_RUN.fullmatch(part):
            tokens.extend(t for t in jieba.cut(part) if t.strip())
        else:
            tokens.extend(part.split())
    return " ".join(tokens)
