"""Markdown 表格的识别、规范化与切分。

两条硬规则，都是为了让表格「切了也还能用」：

1. **每块都重复表头**。切出来的第二块如果没有表头，那几行数字就是无意义的
   裸值——检索到了也读不懂，等于没检索到。
2. **宁可让单行超限，也不切断一行**。强制所有块 ≤ MAX_CHARS 会让长行被从
   中间截断，那是真正的数据报废；超限只是块大了一点。

不是合法 pipe 表的（没有分隔行、或分隔行与表头列数不一致）一律返回 None，
由调用方降级成「保留原文 + 标记未结构化」——不猜、不丢。
"""

import re

from ragv1.ingest import degrade

# 分隔行：整行由 | 与 - : 空白组成，且至少有一个 -
_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")


def is_table_line(line: str) -> bool:
    """该行是否属于表格体（允许前导空白——真实语料里存在缩进的表）。"""
    return line.strip().startswith("|")


def is_separator_line(line: str) -> bool:
    """是否是 Markdown 表的分隔行，形如 | --- | :-- |。"""
    s = line.strip()
    return bool(_SEPARATOR_RE.match(s)) and "-" in s


def split_row(line: str) -> list[str]:
    """把一行拆成单元格。

    不能朴素地 line.split("|")——单元格里的竖线可以写成 \\| 转义，
    朴素切法会凭空多出一列，整张表的列就全错位了。
    """
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]

    cells: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in s:
        if escaped:
            # Markdown 只有 \| 与 \\ 两种转义。其余 \X 保持字面——
            # 表格里到处是 Windows 路径（C:\temp）和正则（\d+），
            # 一律吃掉反斜杠是静默的数据损坏。
            if ch in ("|", "\\"):
                buf.append(ch)
            else:
                buf.append("\\")
                buf.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "|":
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if escaped:
        buf.append("\\")  # 行尾落单的反斜杠，原样保留
    cells.append("".join(buf).strip())
    return cells


def _render_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def normalize_table(raw_lines: list[str]) -> str | None:
    """把原始表格行规范化成 Markdown 表；不是合法表则返回 None。

    合法性只看两件事：第 2 行必须是分隔行；分隔行与表头列数必须一致。
    数据行列数不一致时按表头宽度补齐/截断——这是排版噪声，不是结构错误。
    """
    if len(raw_lines) < 2:
        return None
    if not is_separator_line(raw_lines[1]):
        return None

    header = split_row(raw_lines[0])
    sep = split_row(raw_lines[1])
    if not header or len(header) != len(sep):
        return None

    width = len(header)
    out = [_render_row(header), _render_row(["---"] * width)]
    for raw in raw_lines[2:]:
        cells = split_row(raw)
        cells = (cells + [""] * width)[:width]
        out.append(_render_row(cells))
    return "\n".join(out)


def split_table(md: str, budget: int) -> tuple[list[str], tuple[str, ...]]:
    """把 Markdown 表切成若干块，每块都带表头。

    返回 (块列表, 降级码元组)。单行自身就超过 budget 时，该行**不切**，
    单独成一块并落 table_row_too_long。
    """
    lines = md.splitlines()
    if len(lines) < 2:
        return [md], ()

    header_block = f"{lines[0]}\n{lines[1]}"
    rows = lines[2:]
    if not rows:
        # 只有表头没有数据行：原样一块，不产空块也不抛
        return [md], ()

    parts: list[str] = []
    codes: list[str] = []
    current = header_block
    has_rows = False

    for row in rows:
        if len(f"{current}\n{row}") <= budget:
            current = f"{current}\n{row}"
            has_rows = True
            continue

        if has_rows:
            parts.append(current)

        if len(f"{header_block}\n{row}") > budget:
            # 单行就装不下：整行成块，宁可超限也不切断
            parts.append(f"{header_block}\n{row}")
            codes.append(degrade.TABLE_ROW_TOO_LONG)
            current, has_rows = header_block, False
        else:
            current, has_rows = f"{header_block}\n{row}", True

    if has_rows:
        parts.append(current)

    return (parts or [md]), degrade.normalize(codes)
