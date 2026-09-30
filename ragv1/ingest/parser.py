"""Markdown → ParsedDoc。

保留标题层级（后续按标题切块要用），剥离 YAML frontmatter
（语料里的 frontmatter 是站点元数据，不是正文，也不该被当成标题）。
"""

import re

from ragv1.types import Block, ParsedDoc

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _strip_frontmatter(text: str) -> str:
    """丢弃开头由 --- 行包裹的 frontmatter 块。

    没有闭合的 --- 时整体视为正文，不做剥离——宁可多留，不可误删。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :])
    return text


def parse_document(text: str, doc_id: str) -> ParsedDoc:
    """解析成一组 Block：每个标题一个（含其下正文），标题前的正文为 level=0。"""
    body = _strip_frontmatter(text)

    blocks: list[Block] = []
    level, title = 0, ""
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content or level > 0:
            blocks.append(Block(level=level, title=title, text=content))

    in_fence = False
    for line in body.splitlines():
        # 围栏代码块：``` / ~~~ 之间的内容原样保留，其中的 # 是注释不是标题。
        # 技术文档里代码块遍地都是，误判会产生幽灵 section，让后续块的
        # heading_path 全错，图谱也会长出假节点。
        if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
            in_fence = not in_fence
            buf.append(line)
            continue

        m = None if in_fence else _HEADING_RE.match(line)
        if m:
            flush()
            buf.clear()
            level, title = len(m.group(1)), m.group(2).strip()
        else:
            buf.append(line)
    flush()

    # 丢掉既非标题、又没有正文的空块（例如文档开头的空行）
    return ParsedDoc(
        doc_id=doc_id,
        blocks=tuple(b for b in blocks if b.level > 0 or b.text),
    )
