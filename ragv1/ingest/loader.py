"""文档 → 有序 Element 流。

版面分析在这里落地：把一份文档拆成「正文段 / 表格 / 图片」三类元素，
并给每个元素编上全文档的阅读序号（order）。

**文本连续段**：同一 heading_path 下连续的正文行会攒成一个 text Element，
表格与图片是断开点。这条规则是「纯文本文档产出与改造前逐字节一致」的来源——
把连续的 text 段拼回去，就是原来 chunker._iter_sections 产出的 section 正文。

**图片元素只在配了引擎时才拆出来**（见 `_images_on`）。没配引擎时图片行按
普通正文处理，原始标记 `![alt](url)` 保留在正文里——那是阶段一的行为，
刻意保留：降级态下**原文比 alt 更有信息量**（URL 还在），而拆出元素反而
会把标记吃掉。
"""

import re
from pathlib import Path

from ragv1 import config
from ragv1.ingest import degrade, parser, table
from ragv1.ingest.image_channels import ChannelConfig, resolve_image_element
from ragv1.types import Element

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})

# 整行恰好是一个图片标记（行内图片不算——拆开会丢上下文）
_IMAGE_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<ref>[^)]+)\)$")
# 引用式图片 ![alt][id]：要查文末的定义表才知道指向哪张图，本期不解析
_REF_IMAGE_RE = re.compile(r"^!\[[^\]]*\]\[[^\]]*\]$")


def markdown_elements(
    text: str,
    doc_id: str,
    *,
    ocr=None,
    vlm=None,
    base_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    fetcher=None,
    cfg: ChannelConfig | None = None,
) -> list[Element]:
    """把 Markdown 文本拆成有序 Element 流。

    doc_id 只用于追溯，不参与元素内容的生成。图片相关的参数缺省时，
    图片行按普通正文处理（阶段一行为）。
    """
    channel_cfg = cfg or ChannelConfig()
    # 只有「通道开着」且「至少给了一个引擎」时才拆图片元素——否则拆了也没人处理
    images_on = channel_cfg.enabled and (ocr is not None or vlm is not None)
    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
    cache_dir = Path(cache_dir) if cache_dir is not None else config.IMAGE_CACHE_DIR

    body = parser.strip_frontmatter(text)
    lines = body.splitlines()

    elements: list[Element] = []
    stack: list[str] = []  # 标题栈：从根到当前节的完整路径
    buf: list[str] = []  # 当前正文段的行缓冲
    pending: list[str] = []  # 挂在当前正文段上的降级码
    in_fence = False

    def flush() -> None:
        content = "\n".join(buf).strip()
        codes = degrade.normalize(pending)
        buf.clear()
        pending.clear()
        if content:
            elements.append(
                Element(
                    kind="text",
                    order=len(elements),
                    text=content,
                    heading_path=tuple(stack),
                    degrade=codes,
                )
            )

    i = 0
    while i < len(lines):
        line = lines[i]

        # 围栏代码块：整体原样保留，其中的 # 不是标题、| 不是表格、![]( ) 不是图片
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

            if images_on:
                stripped = line.strip()
                img = _IMAGE_RE.match(stripped)
                if img:
                    flush()
                    element = Element(
                        kind="image",
                        order=len(elements),
                        text=img.group("alt").strip(),
                        heading_path=tuple(stack),
                        image_ref=img.group("ref").strip(),
                    )
                    elements.append(
                        resolve_image_element(
                            element,
                            ocr,
                            vlm,
                            base_dir=base_dir,
                            cache_dir=cache_dir,
                            fetcher=fetcher,
                            cfg=channel_cfg,
                        )
                    )
                    i += 1
                    continue

                if _REF_IMAGE_RE.match(stripped):
                    # 不 continue：这一行照常进正文缓冲，原文得以保留
                    pending.append(degrade.IMAGE_REF_UNRESOLVED)

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


def load_document(path: Path | str, doc_id: str, **engines) -> list[Element]:
    """按扩展名分派。未知后缀返回空列表（不抛异常）。

    图片相关的引擎参数（ocr / vlm / cfg / fetcher）原样透传给 markdown_elements——
    不透传的话，从「文件路径」入口进来时图片永远不被处理，只有直接调
    markdown_elements 才生效。这类「单测绿、端到端静默失效」的缺口要防。

    base_dir 缺省取**文档所在目录**：图片的相对路径是相对文档解析的。
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_SUFFIXES:
        kwargs = dict(engines)
        kwargs.setdefault("base_dir", path.parent)
        return markdown_elements(
            path.read_text(encoding="utf-8"), doc_id=doc_id, **kwargs
        )
    return []
