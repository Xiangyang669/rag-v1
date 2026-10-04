"""MarkdownLoader：文档 → 有序 Element 流。

阶段一只产出 text 与 table 两种元素——图片行继续留在正文里，与改造前一致。
表格是正文的**断开点**：这样表格才能独立成块，不和正文混在一起。
"""

from pathlib import Path

from ragv1.ingest import degrade
from ragv1.ingest.loader import load_document, markdown_elements
from ragv1.ingest.ocr import OcrResult


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


# ─────────────────────────────────────────────────────────────
# 图片元素（阶段二 Task 6）
# ─────────────────────────────────────────────────────────────

# OCR 抽到的字数必须过 OCR_MIN_CHARS（默认 10），否则会被判成「没料」
_OCR_TEXT = "图里的字：服务状态正常"


class _Ocr:
    def extract(self, image):
        return OcrResult(_OCR_TEXT, 0.9, (_OCR_TEXT,))


def _with_ocr(md, tmp_path):
    return markdown_elements(
        md, doc_id="d.md", ocr=_Ocr(), base_dir=tmp_path, cache_dir=tmp_path / "c"
    )


def test_image_line_becomes_image_element(tmp_path):
    (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\nbody")
    md = "# H\n\n前段。\n\n![示意图](a.png)\n\n后段。\n"

    els = _with_ocr(md, tmp_path)

    assert [e.kind for e in els] == ["text", "image", "text"]
    assert _OCR_TEXT in els[1].text
    assert els[1].image_ref == "a.png"
    assert [e.order for e in els] == [0, 1, 2]


def test_image_inside_code_fence_is_not_an_image(tmp_path):
    """代码块里的 ![..](..) 是示例代码，不是图片"""
    md = "# H\n\n```md\n![x](y.png)\n```\n"
    els = _with_ocr(md, tmp_path)
    assert [e.kind for e in els] == ["text"]
    assert "![x](y.png)" in els[0].text


def test_inline_image_stays_in_text(tmp_path):
    """spec §6.1：行内图片不拆开，保留上下文"""
    md = "# H\n\n前面文字 ![x](y.png) 后面文字\n"
    els = _with_ocr(md, tmp_path)
    assert [e.kind for e in els] == ["text"]
    assert "![x](y.png)" in els[0].text


def test_reference_style_image_degrades_and_stays_in_text(tmp_path):
    """引用式图片要查文末定义表，本期不解析——保留原文并落码，不丢"""
    md = "# H\n\n![x][id]\n\n[id]: y.png\n"
    els = _with_ocr(md, tmp_path)
    assert [e.kind for e in els] == ["text"]
    assert degrade.IMAGE_REF_UNRESOLVED in els[0].degrade


def test_no_engines_configured_keeps_phase1_behavior(tmp_path):
    """不传 ocr/vlm 时，图片行仍是普通正文（含原始标记）—— 阶段一测试因此不受影响"""
    md = "# H\n\n![示意图](a.png)\n"
    els = markdown_elements(md, doc_id="d.md")
    assert [e.kind for e in els] == ["text"]
    assert "![示意图](a.png)" in els[0].text


# ─────────────────────────────────────────────────────────────
# PDF（阶段二 Task 8）
# ─────────────────────────────────────────────────────────────


class _C:
    """一个字符（text 可以是多字，够用）。"""

    def __init__(self, text, x0, top, x1, bottom):
        self.text, self.x0, self.top, self.x1, self.bottom = text, x0, top, x1, bottom


class _Page:
    def __init__(self, chars):
        self.chars = list(chars)
        self.images = []

    def find_tables(self):
        return []


class _Pdf:
    def __init__(self, pages):
        self.pages = list(pages)


def test_pdf_elements_returns_text_regions():
    from ragv1.ingest import loader

    els = loader.pdf_elements(_Pdf([_Page([_C("你好", 0, 0, 20, 10)])]), doc_id="a.pdf")
    assert [e.kind for e in els] == ["text"]
    assert "你好" in els[0].text
    assert els[0].page == 1


def test_pdf_page_numbers_are_one_based_and_ordered():
    from ragv1.ingest import loader

    els = loader.pdf_elements(
        _Pdf([_Page([_C("第一页", 0, 0, 20, 10)]), _Page([_C("第二页", 0, 0, 20, 10)])]),
        doc_id="a.pdf",
    )
    assert [e.page for e in els] == [1, 2]
    assert [e.order for e in els] == [0, 1]


def test_pdf_table_region_becomes_table_element():
    from ragv1.ingest import loader

    class _Table:
        bbox = (0, 0, 100, 50)
        rows = [["a", "b"], ["1", "2"]]

    class _TablePage(_Page):
        def find_tables(self):
            return [_Table()]

    # 页上必须有文字层：没有文字层的页会被当成扫描件整页渲染，压根不走表格分支
    page = _TablePage([_C("表", 0, 200, 10, 210)])
    els = loader.pdf_elements(_Pdf([page]), doc_id="a.pdf")

    kinds = [e.kind for e in els]
    assert "table" in kinds
    table = els[kinds.index("table")]
    assert "| a | b |" in table.text
    assert "| --- | --- |" in table.text
    assert table.page == 1


def test_corrupt_pdf_degrades_instead_of_raising(tmp_path):
    """损坏/加密的 PDF 不能抛出"""
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    assert isinstance(load_document(bad, doc_id="bad.pdf"), list)
