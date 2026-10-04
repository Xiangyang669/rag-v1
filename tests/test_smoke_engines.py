"""真实引擎验证（默认跳过）。

单测一律注入假引擎，不碰网络、不花钱；**真实引擎能不能跑通**是另一回事，
由这个文件回答。默认不跑（pytest.ini 里 `-m "not smoke"`），要跑显式来：

    py -m pytest -m smoke -v -s

`-s` 会打印出实际的 OCR 文字与多模态描述——那些是验收证据，别只看绿点。
"""

from pathlib import Path

import pytest

from ragv1.ingest.loader import load_document
from ragv1.ingest.ocr import build_ocr_engine
from ragv1.ingest.vlm import build_vision_engine

pytestmark = pytest.mark.smoke

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.skipif(build_ocr_engine("rapidocr") is None, reason="RapidOCR 不可用")
def test_ocr_reads_text_from_screenshot():
    result = build_ocr_engine("rapidocr").extract(
        (FIXTURES / "ocr_sample.png").read_bytes()
    )
    print("\nOCR 实测输出:", repr(result.text), "置信度", round(result.confidence, 3))
    # OCR 会在中英文之间插空格，比对时去掉
    assert "服务状态正常" in result.text.replace(" ", "")
    assert "CPU" in result.text.replace(" ", "")
    assert result.confidence > 0.5


@pytest.mark.skipif(
    build_vision_engine("siliconflow") is None, reason="未配置 SILICONFLOW_API_KEY"
)
def test_vlm_describes_flowchart():
    vision = build_vision_engine("siliconflow").describe(
        (FIXTURES / "flow.png").read_bytes()
    )
    print("\nVLM 实测输出:", vision)
    assert vision.image_type in {"流程图", "图表"}
    assert vision.content, "多模态没给出任何内容"


def test_pdf_fixture_yields_paged_elements():
    """PDF 入口的端到端：两页都要出来，页码正确。不需要任何外部引擎。"""
    elements = load_document(FIXTURES / "table_doc.pdf", doc_id="table_doc.pdf")
    print(
        "\nPDF 实测元素:",
        [(e.kind, e.page, e.text[:24].replace("\n", " ")) for e in elements],
    )
    pages = {e.page for e in elements}
    assert 1 in pages and 2 in pages
    assert any(e.kind == "table" for e in elements), "fixture 里的表格没被识别"

    # 表格里必须是**文字**——曾经这里装的是单元格 bbox 坐标，只有真跑才看出来
    table = next(e for e in elements if e.kind == "table")
    assert "rrf_k" in table.text and "max_chars" in table.text
    assert "| --- |" in table.text
