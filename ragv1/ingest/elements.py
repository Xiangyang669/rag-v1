"""Element 流 → list[Chunk]。

这是三路**共用**的分块入口（与改造前 chunk_document 同一地位）：向量路、
全文路、图谱路拿到的是同一个列表，块边界与 chunk_id 天然对齐。

与 chunk_document 的关键差别只有一条：**表格走独立分支**。正文段仍旧交给
chunker._group_text（含 OVER_CAP 再切），表格则交给 table.split_table——
后者按「表头 + 若干行」切并保证每块重复表头，绕开了按字符硬切的路径。

「纯文本文档产出与改造前逐字节一致」由两件事共同保证：合并连续正文段
（_merge_text_runs），以及 chunk_id 仍用全局递增的 index（make_chunk_id
的签名没变）。
"""

from dataclasses import replace

from ragv1.ingest import chunker, table
from ragv1.types import Chunk, Element


def _merge_text_runs(elements: list[Element]) -> list[Element]:
    """同一 heading_path 下相邻的 text 元素合并成一个正文段。

    表格/图片是**唯一的断开点**——它们必须独立成块，不能被卷进正文。
    合并回来的正文段与 chunker._iter_sections 产出的 section 正文完全相同，
    这就是纯文本等价性的来源。
    """
    merged: list[Element] = []
    for el in elements:
        if (
            el.kind == "text"
            and merged
            and merged[-1].kind == "text"
            and merged[-1].heading_path == el.heading_path
        ):
            prev = merged[-1]
            merged[-1] = replace(prev, text=f"{prev.text}\n\n{el.text}")
        else:
            merged.append(el)
    return merged


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

    for seg in _merge_text_runs(elements):
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
