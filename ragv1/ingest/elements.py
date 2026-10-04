"""Element 流 → list[Chunk]。

这是三路**共用**的分块入口（与改造前 chunk_document 同一地位）：向量路、
全文路、图谱路拿到的是同一个列表，块边界与 chunk_id 天然对齐。

与 chunk_document 的关键差别只有一条：**表格走独立分支**。正文段仍旧交给
chunker._group_text（含 OVER_CAP 再切），表格则交给 table.split_table——
后者按「表头 + 若干行」切并保证每块重复表头，绕开了按字符硬切的路径。

「纯文本文档产出与改造前逐字节一致」由两件事共同保证：loader 把同一节里
连续的正文行攒成**一个** text Element（与 chunker._iter_sections 的 section
正文同构），以及 chunk_id 仍用全局递增的 index（make_chunk_id 签名没变）。

⚠️ 曾经这里有一个「按 heading_path 相等合并相邻 text 元素」的步骤，是错的：
相邻的 text 元素只可能来自**连续两个同名标题**（那是两个不同的节），中间隔着
表格的两段正文并不会相邻（表格就在它们之间）。所以那个合并只会在纯文本文档
上把两个节错并成一个块，直接违反等价性。删除它，规范由 loader 单独保证。
"""

from ragv1.ingest import chunker, table
from ragv1.types import Chunk, Element


def _emit(
    chunks: list[Chunk],
    doc_id: str,
    seg: Element,
    texts: list[str],
    kind: str,
    codes: tuple[str, ...],
) -> None:
    """把一段切好的文本逐个变成 Chunk。chunk_id 用全局递增的 index。"""
    for part, text in enumerate(texts):
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=chunker.make_chunk_id(doc_id, seg.heading_path, len(chunks)),
                doc_id=doc_id,
                heading_path=seg.heading_path,
                text=text,
                kind=kind,
                page=seg.page,
                order=seg.order,
                part=part,
                bbox=seg.bbox,
                image_ref=seg.image_ref,
                image_path=seg.image_path,
                table_structured=seg.table_structured,
                degrade=codes,
            )
        )


def chunk_elements(elements: list[Element], doc_id: str, max_chars: int) -> list[Chunk]:
    """把有序 Element 流切成三路共用的块列表。"""
    chunks: list[Chunk] = []

    for seg in elements:
        head = chunker._head(seg.heading_path)

        if seg.kind == "table":
            budget = max_chars - len(head)
            if budget <= 0:
                # 标题前缀本身就超限：放弃前缀，保证表格本身可用
                head, budget = "", max_chars
            parts, codes = table.split_table(seg.text, budget)
            _emit(chunks, doc_id, seg, [head + p for p in parts], "table", codes)

        elif seg.kind == "image":
            # 图片整段成块，不切——描述文本被切断就失去了整体含义
            _emit(chunks, doc_id, seg, [head + seg.text], "image", seg.degrade)

        else:
            # 正文：复用原有实现，等价性靠它保证
            texts = chunker._group_text(seg.heading_path, seg.text, max_chars)
            _emit(chunks, doc_id, seg, texts, "text", seg.degrade)

    return chunks
