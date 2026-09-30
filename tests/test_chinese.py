"""中文语料处理。

失败模式（已实测）：FTS5 的 unicode61 分词器把**连续 CJK 串当成一个整
token**，所以整串能匹配、任何子串/分词查询都静默返回 0 条——不报错，
只是召回为空。修法是入库前先分词、查询侧走同一套分词。
"""

import sqlite3

from ragv1.types import Chunk
from ragv1.text.tokenize import tokenize_for_index
from ragv1.store.fts_store import FtsStore

ZH = [Chunk("z1", "d", ("知识图谱",), "知识图谱检索的实现方式")]


def test_raw_unicode61_would_miss_substring():
    """对照实验：不做分词时，子串查询静默返回 0 条。"""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(body, tokenize='unicode61')")
    con.execute("INSERT INTO t(body) VALUES ('知识图谱检索')")
    assert con.execute("SELECT count(*) FROM t WHERE t MATCH '知识图谱'").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM t WHERE t MATCH '知识图谱检索'").fetchone()[0] == 1


def test_tokenize_splits_chinese():
    out = tokenize_for_index("知识图谱检索")
    assert " " in out
    assert "知识图谱检索" != out


def test_tokenize_keeps_ascii_intact():
    out = tokenize_for_index("requests.get timeout")
    assert "requests.get" in out
    assert "timeout" in out


def test_chinese_query_hits(tmp_path):
    s = FtsStore(tmp_path / "kb.db")
    s.add(ZH)
    assert [h.chunk_id for h in s.search("知识图谱", k=5)] == ["z1"]


def test_chinese_chunk_still_keeps_original_text(tmp_path):
    """分词只作用于 FTS 表；回源用的原文不能被改写。"""
    s = FtsStore(tmp_path / "kb.db")
    s.add(ZH)
    assert s.text_of("z1") == ZH[0].text
