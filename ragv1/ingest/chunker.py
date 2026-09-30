"""ParsedDoc → list[Chunk]。

这是三路**共用**的分块（硬约束一）：向量路、全文路、图谱路拿到的是
同一个列表，因此块边界与 chunk_id 天然对齐，融合时不会对不上。

chunk_id 必须是决定性的（同样输入两次运行得到同样的 ID），否则跨重启
无法把检索结果对回原文。

超长节的再切（OVER_CAP）：按标题切完之后，个别 section 仍可能远超上限
（技术文档常见：一大段参数表）。这些节按**段落边界**贪心分组；单段本身
就超限时按字符硬切。分组**只在同一 heading_path 内进行**，且不产出空块。
"""

import hashlib
import re
from typing import Iterator

from ragv1.types import Chunk, ParsedDoc

# heading path 在块文本里的分隔符
PATH_SEP = " > "

# 段落边界：空行（含只含空白的行）
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def make_chunk_id(doc_id: str, heading_path: tuple[str, ...], index: int) -> str:
    """决定性地生成块 ID。index 区分同一 heading 下被切出的多块。"""
    raw = f"{doc_id}|{'/'.join(heading_path)}|{index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _iter_sections(doc: ParsedDoc) -> Iterator[tuple[tuple[str, ...], str]]:
    """按标题栈产出 (heading_path, 正文)。

    维护一个标题栈：遇到 level=N 的标题就弹到 N-1 层再压入，
    因此每节的 heading_path 是「从根到该节」的完整路径。
    正文为空的节不产出。
    """
    stack: list[str] = []
    for block in doc.blocks:
        if block.level > 0:
            while len(stack) >= block.level:
                stack.pop()
            stack.append(block.title)
        body = block.text.strip()
        if body:
            yield tuple(stack), body


def _head(heading_path: tuple[str, ...]) -> str:
    """块文本的 heading path 前缀（含换行）。"""
    prefix = PATH_SEP.join(heading_path)
    return f"{prefix}\n" if prefix else ""


def _paragraphs(text: str) -> list[str]:
    return [p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]


def _pieces_within(text: str, budget: int) -> list[str]:
    """把正文拆成不超过 budget 的片段：优先段落边界，单段过长则硬切。"""
    pieces: list[str] = []
    for para in _paragraphs(text):
        if len(para) <= budget:
            pieces.append(para)
        else:
            pieces.extend(para[i : i + budget] for i in range(0, len(para), budget))
    return pieces


def _group_text(heading_path: tuple[str, ...], body: str, max_chars: int) -> list[str]:
    """产出一节对应的一个或多个块文本，每块长度不超过 max_chars。"""
    head = _head(heading_path)
    budget = max_chars - len(head)
    if budget <= 0:
        # 标题前缀本身就超限：放弃前缀，保证块本身可用
        head, budget = "", max_chars

    if len(body) <= budget:
        return [head + body]

    groups: list[str] = []
    current = ""
    for piece in _pieces_within(body, budget):
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= budget:
            current = candidate
        else:
            if current:
                groups.append(current)
            current = piece
    if current:
        groups.append(current)

    return [head + g for g in groups if g.strip()]


def chunk_document(doc: ParsedDoc, max_chars: int) -> list[Chunk]:
    """按标题层级切块，超长节按 OVER_CAP 再切。

    无标题的文档整体作为一块，heading_path 为空元组。
    """
    chunks: list[Chunk] = []

    for heading_path, body in _iter_sections(doc):
        for text in _group_text(heading_path, body, max_chars):
            if not text.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, heading_path, len(chunks)),
                    doc_id=doc.doc_id,
                    heading_path=heading_path,
                    text=text,
                )
            )

    return chunks


def over_cap_ratio(doc: ParsedDoc, max_chars: int) -> float:
    """切分前，超过长度上限的节占全部非空节的比例。

    给出「多少比例的内容原本超标」这一实测数字，而不是拍脑袋说「文档很碎」。
    """
    sections = list(_iter_sections(doc))
    if not sections:
        return 0.0
    over = sum(
        1 for heading_path, body in sections if len(_head(heading_path)) + len(body) > max_chars
    )
    return over / len(sections)
