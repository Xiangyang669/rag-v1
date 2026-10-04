"""图片双通道：OCR + 多模态，按内容分流，最后合并。

一条图片元素要回答两个不同的问题：

- **图里写了什么字**（扫描件、报错截图、表格截图）→ OCR 通道
- **这张图画的是什么**（流程图、架构图、照片）→ 多模态通道

两条通道**互补**，不是二选一：一张流程图往往既有节点上的文字（OCR 拿得到），
又有那层文字表达不出的关系（多模态拿得到）。所以「两者兼有」时要**都产出**。

## 分流规则表

| OCR 有料¹ | 多模态可用 | 多模态判为视觉类 | 输出 |
|---|---|---|---|
| 是 | 否 | — | 仅 OCR + `vlm_unavailable` |
| 是 | 是 | 否（文字类） | 仅 OCR |
| 是 | 是 | 是 | **OCR + 多模态（都产出）** |
| 否 | 是 | — | 仅多模态 |
| 否 | 否 | — | 占位文本 + `ocr_empty` + `vlm_unavailable` |

¹ 「有料」= 文本长度 ≥ `min_chars` 且置信度 ≥ `min_conf`。

## 降级码的记法（不对称，这是有意的）

- **通道级失败**（引擎不可用 / 抛异常 / 返回坏 JSON）**永远记**——
  它是配置或环境缺陷，使用者需要知道
- **通道正常但没找到料**只在**两条通道都没产出**时才记——
  照片里没有字是图的天然属性，不是缺陷

合并格式（只含非空段）：`[图片] 标题 / 类型 / 内容 / 数据 / 关系 / 文字`。
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from ragv1 import config
from ragv1.ingest import degrade
from ragv1.ingest.image_source import (
    ImageBytes,
    cache_image_bytes,
    resolve_image_bytes,
)
from ragv1.ingest.vlm import VisionResult
from ragv1.types import Element

PLACEHOLDER = "[图片]"

# 多模态判定为「视觉为主」的图片类型。其余（截图/其他/空）按文字为主处理。
_VISUAL_TYPES = frozenset({"图表", "流程图", "照片"})

# 通道状态
_OK = "ok"
_EMPTY = "empty"
_LOW_CONF = "low_conf"
_FAILED = "failed"
_UNAVAILABLE = "unavailable"
_BAD_JSON = "bad_json"


@dataclass(frozen=True)
class ChannelConfig:
    """图片通道的可调项。默认值取自 config，测试里可整包替换。"""

    min_chars: int = config.OCR_MIN_CHARS
    min_conf: float = config.OCR_MIN_CONF
    enabled: bool = config.ENABLE_IMAGE


def _run_ocr(engine, image: bytes, cfg: ChannelConfig) -> tuple[str, str]:
    """返回 (可用文本, 状态)。文本非空当且仅当状态为 _OK。"""
    if engine is None:
        return "", _UNAVAILABLE
    try:
        result = engine.extract(image)
    except Exception:  # noqa: BLE001 —— 降级点
        return "", _FAILED

    text = (result.text or "").strip()
    if not text or len(text) < cfg.min_chars:
        # 太短：不足以断定这是「文字为主的图」。码表里没有更细的码可用，
        # 归入 empty——它表达的正是「没抽到值得用的文字」。
        return "", _EMPTY
    if result.confidence < cfg.min_conf:
        return "", _LOW_CONF
    return text, _OK


def _run_vision(engine, image: bytes) -> tuple[VisionResult | None, str]:
    if engine is None:
        return None, _UNAVAILABLE
    try:
        result = engine.describe(image)
    except Exception:  # noqa: BLE001 —— 降级点
        return None, _FAILED
    if result is None:
        return None, _FAILED

    content = (result.content or "").strip()
    raw = (result.raw or "").strip()
    if not content and raw:
        # 解析失败但拿到了原始返回：拿原文当描述，绝不把一张图变成空白
        return replace(result, content=raw), _BAD_JSON
    return result, _OK


def merge_image_text(
    alt: str, ref: str, ocr_text: str, vision: VisionResult | None
) -> str:
    """按固定段序拼出图片块的正文，只含非空段。"""
    title = (alt or "").strip() or ref
    parts = [f"{PLACEHOLDER} {title}".strip()]
    if vision is not None:
        for label, value in (
            ("类型", vision.image_type),
            ("内容", vision.content),
            ("数据", vision.data),
            ("关系", vision.relations),
        ):
            if (value or "").strip():
                parts.append(f"{label}: {value.strip()}")
    if (ocr_text or "").strip():
        parts.append(f"文字: {ocr_text.strip()}")
    return "\n".join(parts)


def resolve_image_element(
    el: Element,
    ocr,
    vlm,
    *,
    base_dir: Path,
    cache_dir: Path,
    fetcher: Callable[[str], bytes] | None = None,
    cfg: ChannelConfig = ChannelConfig(),
    image_data: bytes | None = None,
) -> Element:
    """把图片元素解析成最终文本元素（kind 仍为 image）。

    任何失败都落码返回，绝不抛——一张图出问题不该让整篇文档入库失败。

    `image_data`：**已经在手上的像素**（PDF 里图片字节就在页对象里，不需要
    「取回」）。给了就直接用，并尽力落进缓存换一个 image_path。
    """
    ref = el.image_ref or ""
    placeholder = f"{PLACEHOLDER} {el.text.strip() or ref}".strip()

    if not cfg.enabled:
        return replace(
            el,
            text=placeholder,
            degrade=degrade.normalize((*el.degrade, degrade.VLM_UNAVAILABLE)),
        )

    if image_data is not None:
        got = ImageBytes(image_data, cache_image_bytes(image_data, ref or "image", cache_dir))
    else:
        got = resolve_image_bytes(ref, base_dir, cache_dir, fetcher=fetcher)
    codes = [*el.degrade, *got.degrade]

    if got.data is None:
        # 拿不到像素：两条通道都无从谈起。只报取图失败——报「通道没料」
        # 会误导（它们根本没跑过）。
        return replace(
            el, text=placeholder, image_path=None, degrade=degrade.normalize(codes)
        )

    ocr_text, ocr_status = _run_ocr(ocr, got.data, cfg)
    vision, vlm_status = _run_vision(vlm, got.data)

    if ocr_text and vision is not None and vision.image_type in _VISUAL_TYPES:
        keep_ocr, keep_vision = ocr_text, vision  # 两者兼有 → 都产出
    elif ocr_text:
        keep_ocr, keep_vision = ocr_text, None  # 文字为主 → 仅 OCR
    elif vision is not None:
        keep_ocr, keep_vision = "", vision  # 视觉为主 → 仅多模态
    else:
        keep_ocr, keep_vision = "", None  # 都没有 → 占位

    # 通道级失败：无论别的通道有没有产出，都要告知
    if ocr_status == _FAILED:
        codes.append(degrade.OCR_FAILED)
    if vlm_status == _UNAVAILABLE:
        codes.append(degrade.VLM_UNAVAILABLE)
    elif vlm_status == _FAILED:
        codes.append(degrade.VLM_FAILED)
    elif vlm_status == _BAD_JSON:
        codes.append(degrade.VLM_BAD_JSON)

    # 通道跑了但没料：只在两条通道都没产出时才记
    if not keep_ocr and keep_vision is None:
        if ocr_status == _LOW_CONF:
            codes.append(degrade.OCR_LOW_CONF)
        elif ocr_status != _FAILED:
            codes.append(degrade.OCR_EMPTY)

    return replace(
        el,
        text=merge_image_text(el.text, ref, keep_ocr, keep_vision),
        image_path=got.path,
        degrade=degrade.normalize(codes),
    )
