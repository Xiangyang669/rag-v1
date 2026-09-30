"""系统内流通的数据形状。

三路召回（向量 / 全文 / 图谱）必须输出同一种 `Hit`，融合层才只有一套逻辑。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """文档中的一个块。

    level > 0 表示这是一行标题（title 为其文本）；level == 0 表示正文段。
    """

    level: int
    title: str
    text: str


@dataclass(frozen=True)
class ParsedDoc:
    doc_id: str
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Chunk:
    """三路共用的检索单元。

    chunk_id 由 (doc_id, heading_path, 序号) 决定性地生成——
    三路拿到的是同一个列表，因此块边界与 ID 天然对齐。
    """

    chunk_id: str
    doc_id: str
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Hit:
    """单路召回结果。path 取 "vector" / "fulltext" / "graph"。"""

    chunk_id: str
    rank: int  # 1-based
    score: float
    path: str


@dataclass(frozen=True)
class FusedHit:
    """融合后的最终结果。

    sources 记录它被哪几条路命中（已排序，保证可复现）——「被三路同时命中」
    本身就是一个强信号，也是对使用者的解释依据。
    """

    chunk_id: str
    rrf_score: float
    sources: tuple[str, ...]
    rank: int  # 1-based
