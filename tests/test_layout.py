"""PDF 版面分析：文字层判定 + 区域聚类。

版面分析**不知道 OCR / 多模态的存在**——它只把一页拆成 text / table / image
三类区域，交给 loader 去决定怎么处理。这样三个模块互不依赖，都能单独替换。

本文件用 pdfplumber Page 的最小替身，不依赖真实 PDF。
"""

from ragv1.ingest import degrade
from ragv1.ingest.layout import analyze_page, has_text_layer


class FakeChar:
    def __init__(self, text, x0, top, x1, bottom):
        self.text, self.x0, self.top, self.x1, self.bottom = text, x0, top, x1, bottom


class FakeTable:
    def __init__(self, bbox, rows):
        self.bbox, self.rows = bbox, rows


class FakePage:
    """只暴露 analyze_page 用到的三个属性。"""

    def __init__(self, chars=(), images=(), tables=()):
        self.chars = list(chars)
        self.images = list(images)
        self._tables = list(tables)

    def find_tables(self):
        return self._tables


def test_chars_may_be_dicts_like_real_pdfplumber():
    """pdfplumber 的 `page.chars` 元素是 **dict**，不是带属性的对象。

    替身用对象来写时看不出这个差别——拿真实 PDF 跑就 AttributeError。
    两种形态都要支持。
    """
    page = FakePage(
        chars=[{"text": "你", "x0": 0, "top": 0, "x1": 10, "bottom": 10}]
    )
    regions = analyze_page(page, 1)
    assert [r.kind for r in regions] == ["text"]
    assert "你" in regions[0].text


def test_table_text_comes_from_extract_not_cells():
    """pdfplumber 的 `TableRow.cells` 存的是**单元格 bbox**，不是文字。

    取文字必须走 `Table.extract()`——用 `.cells` 会把坐标当成表格内容。
    """
    class RealishTable:
        bbox = (0, 0, 100, 50)
        rows = [type("R", (), {"cells": [(0, 0, 10, 10), (10, 0, 20, 10)]})()]

        def extract(self):
            return [["参数", "默认值"], ["k", "10"]]

    page = FakePage(
        chars=[FakeChar("文", 0, 200, 10, 210)], tables=[RealishTable()]
    )
    regions = analyze_page(page, 1)
    table = next(r for r in regions if r.kind == "table")
    assert table.table_rows == (("参数", "默认值"), ("k", "10"))


def test_has_text_layer_false_on_scanned_page():
    assert has_text_layer(FakePage(chars=[])) is False


def test_has_text_layer_true_when_chars_present():
    assert has_text_layer(FakePage(chars=[FakeChar("a", 0, 0, 1, 1)])) is True


def test_scanned_page_is_rendered_and_marked():
    """无文字层的页（扫描件）整页走渲染"""
    regions = analyze_page(FakePage(), 1, render_page=lambda: b"\x89PNG-page")
    assert len(regions) == 1
    assert regions[0].kind == "image" and regions[0].image_bytes == b"\x89PNG-page"
    assert degrade.PDF_PAGE_NO_TEXT_LAYER in regions[0].degrade


def test_render_failure_yields_placeholder_not_silence():
    """渲染失败也不能静默丢页"""

    def boom():
        raise RuntimeError("渲染挂了")

    regions = analyze_page(FakePage(), 1, render_page=boom)
    assert len(regions) == 1
    assert degrade.PDF_RENDER_FAILED in regions[0].degrade


def test_text_page_yields_text_region():
    page = FakePage(chars=[FakeChar("h", 0, 0, 10, 10), FakeChar("i", 11, 0, 20, 10)])
    regions = analyze_page(page, 1)
    assert [r.kind for r in regions] == ["text"]
    assert "hi" in regions[0].text
    assert regions[0].page == 1


def test_table_region_is_separated_from_text():
    page = FakePage(
        chars=[FakeChar("正", 0, 100, 10, 110), FakeChar("文", 11, 100, 20, 110)],
        tables=[FakeTable((0, 0, 100, 50), [["a", "b"], ["1", "2"]])],
    )
    regions = analyze_page(page, 1)
    assert sorted(r.kind for r in regions) == ["table", "text"]
    table = next(r for r in regions if r.kind == "table")
    assert table.table_rows == (("a", "b"), ("1", "2"))


def test_image_region_is_separated():
    page = FakePage(
        chars=[FakeChar("字", 0, 200, 10, 210)],
        images=[{"x0": 0, "top": 0, "x1": 50, "bottom": 50, "stream": b"IMG"}],
    )
    regions = analyze_page(page, 1)
    imgs = [r for r in regions if r.kind == "image"]
    assert len(imgs) == 1
    assert imgs[0].image_bytes is not None or degrade.PDF_RENDER_FAILED in imgs[0].degrade


def test_regions_are_ordered_top_to_bottom():
    page = FakePage(chars=[FakeChar("下", 0, 500, 10, 510), FakeChar("上", 0, 10, 10, 20)])
    texts = [r.text for r in analyze_page(page, 1) if r.kind == "text"]
    assert "".join(texts).startswith("上")


def _line(text, top, height=12.0, width=10.0):
    """一行文字 → 一串 FakeChar（每个字一个字符）。"""
    return [
        FakeChar(ch, i * width, top, (i + 1) * width, top + height)
        for i, ch in enumerate(text)
    ]


def test_nearby_lines_merge_into_one_paragraph():
    """相邻行（行距小）合成一个区域——否则一页 PDF 会碎成几十个块。"""
    page = FakePage(chars=[*_line("第一行", 0), *_line("第二行", 14)])
    regions = [r for r in analyze_page(page, 1) if r.kind == "text"]
    assert len(regions) == 1
    assert "第一行" in regions[0].text
    assert "第二行" in regions[0].text


def test_distant_lines_become_separate_paragraphs():
    """行距大 = 换段，不能被合并"""
    page = FakePage(chars=[*_line("上段", 0), *_line("下段", 200)])
    regions = [r for r in analyze_page(page, 1) if r.kind == "text"]
    assert len(regions) == 2
    assert regions[0].text == "上段" and regions[1].text == "下段"
