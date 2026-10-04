"""PDF 版面分析：把一页拆成 text / table / image 三类区域。

**这是自研的轻量版，不是 ML 版面模型。** 用 pdfplumber 已有的三样东西做区域
聚类：`page.chars`（每个字符带 bbox）、`page.images`、`page.find_tables()`。

为什么不引入 ML 模型（PP-StructureV3 / DocLayout-YOLO）：它们要拖 paddle 或
torch、模型几百 MB、Windows 上装 paddle 常年出问题——与「默认选一个能跑通的」
直接冲突。**已知局限（如实记录）**：双栏/多栏排版的聚类可能切歪。接口保持
最小，以后想换 ML 实现不用动上下游。

本模块**不知道 OCR / 多模态的存在**：它只产出区域，由 loader 决定怎么处理。
"""

from dataclasses import dataclass
from typing import Callable

from ragv1.ingest import degrade

# 同一行的判定：top 差值不超过行高的一半（再给 1 点容差防止行高为 0）
_LINE_TOL_RATIO = 0.5
_LINE_TOL_MIN = 1.0

# 换段的判定：本行 top 距上一行底边超过 1.5 倍行高，就算新段落。
# 行距通常 2-5 点、行高 10-12 点，所以段内行会合并、段间会断开。
_PARAGRAPH_GAP_RATIO = 1.5


@dataclass(frozen=True)
class Region:
    """页面上的一块区域。三种 kind 的载荷同表，用 None/空值区分。"""

    kind: str  # text | table | image
    page: int  # 从 1 计
    bbox: tuple[float, float, float, float]  # (x0, top, x1, bottom)
    text: str = ""
    table_rows: tuple[tuple[str, ...], ...] = ()
    image_bytes: bytes | None = None
    degrade: tuple[str, ...] = ()


def has_text_layer(page) -> bool:
    """该页是否有文字层。

    只看「有没有字符」——空白字符也算文字层，质量判定交给后续（OCR 置信度、
    字符数阈值）。这里不做阈值，是因为阈值属于通道的事，不属于版面分析。
    """
    return bool(page.chars)


def _field(item, name: str):
    """按名字取一个版面对象的字段。

    ⚠️ pdfplumber 的各路对象形态不统一：`page.chars` 的元素是 **dict**，
    `page.images` 也是 dict，而 `find_tables()` 返回的是对象。这里统一兜住，
    否则「替身写成对象」的单测全绿、一跑真实 PDF 就 AttributeError。
    """
    if isinstance(item, dict):
        return item[name]
    return getattr(item, name)


def _bbox_of(item) -> tuple[float, float, float, float]:
    """取一个版面对象的 bbox。

    三种形态都要能处理：pdfplumber 的图片是 dict（x0/top/x1/bottom 键）、
    表格是对象且带 `.bbox` 四元组、字符是对象且只有 `.x0/.top/...` 属性。
    顺序统一为 (x0, top, x1, bottom)——与 pdfplumber 一致。
    """
    if isinstance(item, dict):
        return (
            float(item["x0"]),
            float(item["top"]),
            float(item["x1"]),
            float(item["bottom"]),
        )
    bbox = getattr(item, "bbox", None)
    if bbox is not None:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (float(item.x0), float(item.top), float(item.x1), float(item.bottom))


def _center_inside(box, region_box) -> bool:
    x0, top, x1, bottom = box
    cx, cy = (x0 + x1) / 2.0, (top + bottom) / 2.0
    rx0, rtop, rx1, rbottom = region_box
    return rx0 <= cx <= rx1 and rtop <= cy <= rbottom


def _lines_of(chars) -> list[list]:
    """把字符按 top 聚成行（行内按 x 排序）。"""
    ordered = sorted(chars, key=lambda c: (float(_field(c, "top")), float(_field(c, "x0"))))
    lines: list[list] = []
    current: list = []
    ref_top = None
    ref_height = 0.0

    for ch in ordered:
        top, bottom = float(_field(ch, "top")), float(_field(ch, "bottom"))
        height = max(bottom - top, 0.0)
        tol = max(ref_height * _LINE_TOL_RATIO, _LINE_TOL_MIN)
        if not current or abs(top - ref_top) <= tol:
            if not current:
                ref_top, ref_height = top, height
            current.append(ch)
        else:
            lines.append(current)
            current, ref_top, ref_height = [ch], top, height
    if current:
        lines.append(current)

    for line in lines:
        line.sort(key=lambda c: float(_field(c, "x0")))
    return lines


def _line_text(line) -> str:
    return "".join(str(_field(c, "text")) for c in line)


def _line_box(line) -> tuple[float, float, float, float]:
    return (
        min(float(_field(c, "x0")) for c in line),
        min(float(_field(c, "top")) for c in line),
        max(float(_field(c, "x1")) for c in line),
        max(float(_field(c, "bottom")) for c in line),
    )


def _table_rows_of(table) -> tuple[tuple[str, ...], ...]:
    """取表格的行列**文字**。

    ⚠️ 必须走 `Table.extract()`。pdfplumber 的 `TableRow.cells` 存的是单元格的
    **bbox 四元组**，不是文字——拿它当内容会把坐标拼进 chunk。
    替身/简化结构没有 extract()，此时才退回按序列处理 rows。
    """
    extract = getattr(table, "extract", None)
    if callable(extract):
        try:
            rows = extract()
        except Exception:  # noqa: BLE001 —— 抽取失败就走降级，别让整页崩掉
            rows = None
        if rows:
            return tuple(
                tuple("" if c is None else str(c).strip() for c in row) for row in rows
            )

    rows = []
    for row in getattr(table, "rows", ()) or ():
        cells = getattr(row, "cells", row)
        rows.append(tuple("" if c is None else str(c) for c in cells))
    return tuple(rows)


def _render_failed(page_no: int) -> Region:
    return Region(
        kind="text",
        page=page_no,
        bbox=(0.0, 0.0, 0.0, 0.0),
        degrade=(degrade.PDF_RENDER_FAILED,),
    )


def analyze_page(
    page, page_no: int, render_page: Callable[[], bytes] | None = None
) -> list[Region]:
    """一页 → 按阅读顺序（top 升序）排列的区域列表。"""
    if not has_text_layer(page):
        # 扫描件：整页当一张图，交给图片双通道
        if render_page is None:
            return [
                Region(
                    kind="text",
                    page=page_no,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    degrade=(degrade.PDF_PAGE_NO_TEXT_LAYER,),
                )
            ]
        try:
            data = render_page()
        except Exception:  # noqa: BLE001 —— 降级点：渲染失败不许静默丢页
            return [_render_failed(page_no)]
        return [
            Region(
                kind="image",
                page=page_no,
                bbox=(0.0, 0.0, 0.0, 0.0),
                image_bytes=data,
                degrade=(degrade.PDF_PAGE_NO_TEXT_LAYER,),
            )
        ]

    regions: list[Region] = []

    table_boxes: list[tuple[tuple[float, float, float, float], object]] = []
    for table in page.find_tables() or ():
        box = _bbox_of(table)
        rows = _table_rows_of(table)
        table_boxes.append((box, table))
        regions.append(
            Region(
                kind="table",
                page=page_no,
                bbox=box,
                table_rows=rows,
                degrade=() if rows else (degrade.PDF_TABLE_BBOX_MISSING,),
            )
        )

    image_boxes: list[tuple[float, float, float, float]] = []
    for image in getattr(page, "images", ()) or ():
        box = _bbox_of(image)
        stream = image.get("stream") if isinstance(image, dict) else getattr(image, "stream", None)
        image_boxes.append(box)
        regions.append(
            Region(
                kind="image",
                page=page_no,
                bbox=box,
                image_bytes=stream,
                degrade=() if stream else (degrade.PDF_RENDER_FAILED,),
            )
        )

    absorbed = [box for box, _ in table_boxes] + image_boxes

    # 正文：先聚行，把落在表格/图片区域内的行剔掉，再把相邻行合并成段落
    blocks: list[tuple[tuple[float, float, float, float], list[str]]] = []
    for line in _lines_of(page.chars):
        box = _line_box(line)
        if any(_center_inside(box, region) for region in absorbed):
            continue
        text = _line_text(line)
        if not text.strip():
            continue
        if blocks:
            prev_box, prev_lines = blocks[-1]
            gap = box[1] - prev_box[3]
            if gap <= _PARAGRAPH_GAP_RATIO * max(prev_box[3] - prev_box[1], 0.0):
                blocks[-1] = (
                    (
                        min(prev_box[0], box[0]),
                        prev_box[1],
                        max(prev_box[2], box[2]),
                        max(prev_box[3], box[3]),
                    ),
                    [*prev_lines, text],
                )
                continue
        blocks.append((box, [text]))

    for box, lines in blocks:
        regions.append(
            Region(kind="text", page=page_no, bbox=box, text="\n".join(lines))
        )

    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    return regions
