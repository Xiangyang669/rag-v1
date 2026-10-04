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
class Element:
    """版面分析产出的有序元素——分块之前、结构还完整时的中间表示。

    三种 kind（text / table / image）的载荷放在同一张扁平表里，用 None 区分。
    比给每种 kind 单开一个类好读，也与 Block / Chunk 的风格一致。

    order 是全文档的阅读序号：图片处理可以并行、表格可以按行重排，
    最后都按 order 排回原位，这就是「恢复阅读顺序」的实现。
    """

    kind: str  # text | table | image
    order: int  # 全文档阅读顺序，0 起
    text: str  # text=正文 / table=Markdown 表 / image=最终描述文本

    heading_path: tuple[str, ...] = ()
    page: int | None = None  # PDF 从 1 计；Markdown 为 None
    bbox: tuple[float, float, float, float] | None = None  # (x0, top, x1, bottom)

    # ── image 专有 ──
    image_ref: str | None = None  # 原图 URL / 相对路径
    image_path: str | None = None  # 本地缓存路径

    # ── table 专有 ──
    table_structured: bool = True  # False = 降级为保留原文

    # ── 降级 ──
    degrade: tuple[str, ...] = ()  # 原因码，已排序去重


@dataclass(frozen=True)
class Chunk:
    """三路共用的检索单元。

    chunk_id 由 (doc_id, heading_path, 序号) 决定性地生成——
    三路拿到的是同一个列表，因此块边界与 ID 天然对齐。

    来源元数据（page / order / bbox / image_ref …）全部带默认值，
    因此「只给四个必需字段」的构造方式仍然有效。
    """

    chunk_id: str
    doc_id: str
    heading_path: tuple[str, ...]
    text: str

    # ── 来源元数据 ──
    kind: str = "text"  # text | table | image
    page: int | None = None  # PDF 页码；Markdown 为 None
    order: int = 0  # 全文档阅读序号
    part: int = 0  # 同一元素被切成多块时的序号（0 起）
    bbox: tuple[float, float, float, float] | None = None  # 位置（Markdown 为 None）
    image_ref: str | None = None  # 原图 URL / 相对路径
    image_path: str | None = None  # 本地缓存路径
    table_structured: bool = True  # False = 未结构化降级
    degrade: tuple[str, ...] = ()  # 降级原因码


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
