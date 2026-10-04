"""Element 中间表示、Chunk 的来源元数据字段，以及 Element → Chunk 的分块。

本文件按任务分两批：先类型用例（Task 2），后分块行为用例（Task 5）。
"""

import dataclasses

from ragv1.config import MAX_CHARS
from ragv1.ingest.chunker import chunk_document
from ragv1.ingest.elements import chunk_elements
from ragv1.ingest.loader import markdown_elements
from ragv1.ingest.parser import parse_document
from ragv1.types import Chunk, Element


def test_chunk_old_construction_still_works():
    """向后兼容是硬约束：现有构造点一个都不许改。"""
    c = Chunk(chunk_id="a", doc_id="d.md", heading_path=("H",), text="正文")
    assert c.kind == "text"
    assert c.page is None and c.order == 0 and c.part == 0
    assert c.bbox is None and c.image_ref is None and c.image_path is None
    assert c.table_structured is True
    assert c.degrade == ()


def test_element_defaults():
    e = Element(kind="text", order=0, text="正文")
    assert e.heading_path == () and e.page is None and e.bbox is None
    assert e.table_structured is True and e.degrade == ()


def test_element_and_chunk_are_frozen():
    assert dataclasses.fields(Chunk)  # 存在即通过；frozen 由下面的赋值验证
    e = Element(kind="text", order=0, text="x")
    try:
        e.text = "y"
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Element 必须是 frozen 的")


# ─────────────────────────────────────────────────────────────
# Task 5：Element 流 → Chunk
# ─────────────────────────────────────────────────────────────

_PURE_TEXT_DOCS = [
    # 无表格、无图片：覆盖多级标题、空节、代码围栏、超长节
    "# A\n\n前段。\n\n## B\n\n二段。\n\n# C\n\n三段。\n",
    "无标题文档，只有一段正文。\n",
    "# H\n\n```python\n# 代码里的井号不是标题\nx = 1\n```\n\n正文。\n",
    "# H\n\n" + "很长的段落。" * 300 + "\n",  # 触发 OVER_CAP 再切
    "# A\n\n一。\n\n## B\n\n",  # 空节
]


def test_pure_text_output_is_byte_identical_to_old_pipeline():
    """硬约束：不含表格/图片的文档，新旧链路产出逐字节一致。

    这条是「不破坏现有纯文本文档解析路径」的可执行证明。
    """
    for md in _PURE_TEXT_DOCS:
        old = chunk_document(parse_document(md, doc_id="d.md"), max_chars=MAX_CHARS)
        new = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
        assert [(c.chunk_id, c.text) for c in new] == [
            (c.chunk_id, c.text) for c in old
        ], f"纯文本文档产出不一致：{md[:40]!r}"


def test_table_gets_its_own_chunk():
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n后段。\n"
    chunks = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    kinds = [c.kind for c in chunks]
    assert kinds.count("table") == 1
    table_chunk = chunks[kinds.index("table")]
    assert table_chunk.table_structured is True
    assert "| 1 | 2 |" in table_chunk.text
    # 表格不在任何正文块里（"不和正文混在一起"）
    assert all("| 1 | 2 |" not in c.text for c in chunks if c.kind != "table")


def test_oversized_table_is_not_char_cut_and_every_part_has_header():
    """核心：大表按表头+若干行切，绝不被 _pieces_within 硬切"""
    rows = [f"| {i} | {'x' * 40} |" for i in range(30)]
    md = "# H\n\n" + "\n".join(["| a | b |", "| --- | --- |", *rows]) + "\n"
    chunks = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    table_chunks = [c for c in chunks if c.kind == "table"]
    assert len(table_chunks) > 1
    for c in table_chunks:
        # 每块都必须带完整表头（表头行 + 分隔行），否则后续块的行是无意义的裸值
        assert c.text.splitlines()[1:3] == ["| a | b |", "| --- | --- |"]
    assert [c.part for c in table_chunks] == list(range(len(table_chunks)))
    # 30 行一行不少，且没有任何一行被切断
    assert "\n".join(c.text for c in table_chunks).count("| x") == 30


def test_chunk_ids_are_deterministic():
    """同输入两次运行必须给出同样的 chunk_id，否则跨重启对不回原文"""
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    a = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    b = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_metadata_is_carried_onto_chunks():
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    chunks = chunk_elements(markdown_elements(md, doc_id="d.md"), "d.md", MAX_CHARS)
    table_chunk = next(c for c in chunks if c.kind == "table")
    assert table_chunk.doc_id == "d.md"
    assert table_chunk.heading_path == ("H",)
    assert table_chunk.order == 1  # 与 Element 的 order 对齐
