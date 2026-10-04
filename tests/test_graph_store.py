"""图谱支线的测试。

两个不变量：
- 实体全部从**结构**（heading_path）抽出，不调 LLM —— 因此结果必须决定性可复现
- 每个实体必须挂 posting list（出现在哪些块），否则图谱召回回不到原文
"""

import pytest

from ragv1.types import Chunk
from ragv1.ingest.extract import extract
from ragv1.store.graph_store import GraphStore

DOC = """# requests 模块

## requests.get(url, timeout=None)

发起 GET 请求。

## requests.post(url, data=None)

发起 POST 请求。
"""


@pytest.fixture
def chunks():
    return [
        Chunk("k1", "d", ("requests 模块",), DOC),
        Chunk(
            "k2",
            "d",
            ("requests 模块", "requests.get(url, timeout=None)"),
            "发起 GET 请求。",
        ),
    ]


def test_extracts_function_and_params(chunks):
    entities, relations = extract(chunks)
    kinds = {e.canonical: e.kind for e in entities}
    assert kinds.get("requests.get") == "function"
    assert "timeout" in {e.canonical for e in entities}


def test_has_parameter_relation(chunks):
    _, relations = extract(chunks)
    assert any(
        r.kind == "has_parameter" and r.src == "requests.get" and r.dst == "timeout"
        for r in relations
    )


def test_entity_posting_list(chunks, tmp_path):
    g = GraphStore(tmp_path / "g.db")
    g.build(chunks)
    assert "k2" in g.chunks_of("requests.get")


def test_params_of(chunks, tmp_path):
    g = GraphStore(tmp_path / "g.db")
    g.build(chunks)
    assert "timeout" in g.params_of("requests.get")


def test_alias_resolution(chunks, tmp_path):
    g = GraphStore(tmp_path / "g.db")
    g.build(chunks)
    assert g.aliases().get("get") == "requests.get"


def test_extraction_is_deterministic(chunks):
    a = extract(chunks)
    b = extract(chunks)
    assert [(e.canonical, e.kind) for e in a[0]] == [(e.canonical, e.kind) for e in b[0]]
    assert [(r.src, r.dst, r.kind) for r in a[1]] == [(r.src, r.dst, r.kind) for r in b[1]]


def test_doc_without_headings_yields_no_entities():
    entities, relations = extract([Chunk("x", "d", (), "一段普通正文。")])
    assert entities == [] and relations == []


# ── 真数据暴露的问题：普通标题里的括号被当成函数签名 ──────────
# 合成测试用的 requests.get(url, timeout=None) 无歧义，但真实文档里
# 满是「Default Mode (Beta)」「Browser (Chrome / Firefox)」这类标题。


def test_parenthetical_heading_is_not_a_function():
    chunks = [Chunk("k1", "d", ("Documentation", "Default Mode (Beta)"), "说明")]
    entities, _ = extract(chunks)
    kinds = {e.canonical: e.kind for e in entities}
    assert kinds.get("Default Mode (Beta)") == "section"
    assert "Documentation.Default Mode" not in kinds


def test_non_identifier_params_heading_is_not_a_function():
    chunks = [Chunk("k1", "d", ("Docs", "Browser (Chrome / Firefox)"), "说明")]
    entities, _ = extract(chunks)
    kinds = {e.canonical: e.kind for e in entities}
    assert kinds.get("Browser (Chrome / Firefox)") == "section"
    assert "Docs.Browser" not in kinds
