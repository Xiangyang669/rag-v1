"""MarkdownLoader：文档 → 有序 Element 流。

阶段一只产出 text 与 table 两种元素——图片行继续留在正文里，与改造前一致。
表格是正文的**断开点**：这样表格才能独立成块，不和正文混在一起。
"""

from pathlib import Path

from ragv1.ingest.loader import load_document, markdown_elements


def test_headings_set_heading_path_and_text_is_split_at_table():
    md = "# H\n\n前段。\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n后段。\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text", "table", "text"]
    assert els[0].heading_path == ("H",)
    assert els[0].text == "前段。"
    assert els[1].text.startswith("| a | b |")
    assert els[2].text == "后段。"
    assert [e.order for e in els] == [0, 1, 2]


def test_table_inside_code_fence_is_not_a_table():
    """代码块里的 | 行不是表格——技术文档里代码块遍地，误判会把代码挖走"""
    md = "# H\n\n```bash\n| a | b |\n| --- | --- |\n| 1 | 2 |\n```\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]
    assert "| --- | --- |" in els[0].text


def test_image_line_stays_in_text_this_phase():
    """阶段一边界：图片行不识别，继续留在正文里"""
    md = "# H\n\n![alt](http://x/y.png)\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]
    assert "![alt](http://x/y.png)" in els[0].text


def test_frontmatter_is_stripped():
    md = "---\ntitle: X\n---\n# H\n\n正文。\n"
    els = markdown_elements(md, doc_id="d.md")
    assert "title: X" not in els[0].text
    assert els[0].heading_path == ("H",)


def test_multiple_headings_produce_nested_paths():
    md = "# A\n\n一。\n\n## B\n\n二。\n\n# C\n\n三。\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.heading_path for e in els] == [("A",), ("A", "B"), ("C",)]


def test_malformed_table_degrades_but_keeps_raw_text():
    """畸形表降级保留原文，不丢"""
    md = "# H\n\n| a | b |\n|---|\n| 1 | 2 |\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]  # 没被当成表格
    assert "| 1 | 2 |" in els[0].text


def test_load_document_dispatches_by_suffix(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("# H\n\n正文。\n", encoding="utf-8")
    assert len(load_document(p, doc_id="a.md")) == 1


def test_load_document_unknown_suffix_returns_empty(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")
    assert load_document(p, doc_id="a.json") == []
