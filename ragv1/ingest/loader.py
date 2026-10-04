"""文档 → 有序 Element 流。

版面分析在这里落地：把一份文档拆成「正文段 / 表格 / 图片」三类元素，
并给每个元素编上全文档的阅读序号（order）。

**文本连续段**：同一 heading_path 下连续的正文行会攒成一个 text Element，
表格是唯一的断开点。这条规则是「纯文本文档产出与改造前逐字节一致」的来源——
把连续的 text 段拼回去，就是原来 chunker._iter_sections 产出的 section 正文。

阶段一只产出 text 与 table 两种元素，图片行继续留在正文里（与改造前一致）；
图片元素识别与双通道在阶段二接入。
"""

from pathlib import Path

from ragv1.ingest import parser, table
from ragv1.types import Element

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})


def markdown_elements(text: str, doc_id: str) -> list[Element]:
    """把 Markdown 文本拆成有序 Element 流。

    doc_id 只用于追溯，不参与元素内容的生成。
    """
    body = parser.strip_frontmatter(text)
    lines = body.splitlines()

    elements: list[Element] = []
    stack: list[str] = []  # 标题栈：从根到当前节的完整路径
    buf: list[str] = []  # 当前正文段的行缓冲
    in_fence = False

    def flush() -> None:
        content = "\n".join(buf).strip()
        buf.clear()
        if content:
            elements.append(
                Element(
                    kind="text",
                    order=len(elements),
                    text=content,
                    heading_path=tuple(stack),
                )
            )

    i = 0
    while i < len(lines):
        line = lines[i]

        # 围栏代码块：整体原样保留，其中的 # 不是标题、| 不是表格
        if parser.is_fence_line(line):
            in_fence = not in_fence
            buf.append(line)
            i += 1
            continue

        if not in_fence:
            m = parser._HEADING_RE.match(line)
            if m:
                flush()
                level, title = len(m.group(1)), m.group(2).strip()
                # 与 chunker._iter_sections 同一套弹栈规则，保证 heading_path 一致
                while len(stack) >= level:
                    stack.pop()
                stack.append(title)
                i += 1
                continue

            # 表格起点：本行像表行，且下一行是分隔行
            if (
                table.is_table_line(line)
                and i + 1 < len(lines)
                and table.is_separator_line(lines[i + 1])
            ):
                j = i
                while j < len(lines) and table.is_table_line(lines[j]):
                    j += 1
                raw = lines[i:j]
                normalized = table.normalize_table(raw)
                if normalized is None:
                    # 畸形表：原样并回正文，不猜结构也不丢内容
                    buf.extend(raw)
                else:
                    flush()
                    elements.append(
                        Element(
                            kind="table",
                            order=len(elements),
                            text=normalized,
                            heading_path=tuple(stack),
                        )
                    )
                i = j
                continue

        buf.append(line)
        i += 1

    flush()
    return elements


def load_document(path: Path | str, doc_id: str) -> list[Element]:
    """按扩展名分派。未知后缀返回空列表（不抛异常，也不静默吞掉已知类型）。

    阶段二会在这里加 `.pdf` 分支。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        return markdown_elements(path.read_text(encoding="utf-8"), doc_id=doc_id)
    return []
