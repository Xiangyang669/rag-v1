"""生成阶段二的测试 fixture（三份），落进 tests/fixtures/。

产物随仓库提交；本脚本只在**需要重新生成**时跑。

- ocr_sample.png  带文字的截图（验证 OCR 通道）
- flow.png        流程图（验证多模态通道）
- table_doc.pdf   带文字层与表格的两页 PDF（验证 PDF 入口）

reportlab **只在这个脚本里用**，不进 requirements.txt——它是生成工具，
不是运行期依赖。Pillow 本来就在依赖里。

用法：
    py -m pip install reportlab
    py scripts/make_fixtures.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"

# 中文字体：simhei.ttf 是纯 TTF，Pillow 与 reportlab 都能直接用
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
)


def _font_path() -> Path:
    for candidate in FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise SystemExit("找不到中文字体（试过 simhei.ttf / msyh.ttc）")


def make_ocr_sample(path: Path, font_path: Path) -> None:
    """白底黑字的一张「截图」。字要够大，否则 OCR 容易漏。"""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (720, 180), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 34)
    draw.text((24, 30), "服务状态正常", fill="black", font=font)
    draw.text((24, 90), "2026-10-04  CPU 87%", fill="black", font=font)
    image.save(path)


def make_flowchart(path: Path, font_path: Path) -> None:
    """三个方框 + 箭头，用来验证多模态能不能讲出「流程走向」。"""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (760, 320), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 28)

    boxes = [(40, 120, 200, 200), (300, 120, 460, 200), (560, 120, 720, 200)]
    labels = ["开始", "处理", "结束"]
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        draw.rectangle([x0, y0, x1, y1], outline="black", width=3)
        draw.text((x0 + 44, y0 + 18), label, fill="black", font=font)

    for left, right in zip(boxes, boxes[1:]):
        y = (left[1] + left[3]) // 2
        draw.line([left[2], y, right[0] - 14, y], fill="black", width=3)
        draw.polygon(
            [(right[0], y), (right[0] - 16, y - 9), (right[0] - 16, y + 9)],
            fill="black",
        )

    image.save(path)


def make_table_pdf(path: Path, font_path: Path) -> None:
    """两页：第 1 页有标题 + 正文 + 带边框的表格，第 2 页只有一段文字。

    第 2 页用来验证 page 字段确实是 1 和 2。
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdfmetrics.registerFont(TTFont("CJK", str(font_path)))
    style = ParagraphStyle("cjk", fontName="CJK", fontSize=12, leading=20)

    doc = SimpleDocTemplate(str(path), pagesize=A4, title="检索参数速查")
    rows = [
        ["参数", "默认值", "说明"],
        ["k", "10", "每路召回条数"],
        ["rrf_k", "60", "RRF 的平滑常数"],
        ["max_chars", "1200", "单块字符上限"],
    ]
    table = Table(rows, colWidths=[110, 90, 220])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "CJK"),
                ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ]
        )
    )

    doc.build(
        [
            Paragraph("检索参数速查", style),
            Spacer(1, 12),
            Paragraph("本文汇总三层检索系统里可以调的参数。", style),
            Spacer(1, 12),
            table,
            PageBreak(),
            Paragraph("第二页只有这一段文字，用来验证页码字段。", style),
        ]
    )


def main() -> int:
    font_path = _font_path()
    FIXTURES.mkdir(parents=True, exist_ok=True)

    make_ocr_sample(FIXTURES / "ocr_sample.png", font_path)
    make_flowchart(FIXTURES / "flow.png", font_path)
    make_table_pdf(FIXTURES / "table_doc.pdf", font_path)

    for name in ("ocr_sample.png", "flow.png", "table_doc.pdf"):
        p = FIXTURES / name
        print(f"  {name:18s} {p.stat().st_size:>8,} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
