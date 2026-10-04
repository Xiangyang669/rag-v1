"""OCR 通道：把图片里的**文字**抽出来。

适用场景是扫描件、报错截图、文档截图这类「内容本来就是字」的图。
视觉为主的内容（流程图、照片）交给多模态通道，两条通道可以都产出。

两条设计约束：

- **惰性加载**：rapidocr 会加载 onnx 模型，import 期就加载会让每次启动
  都付几秒代价。首次 extract 才构造。
- **只管抽字，不管降级码**：抽不到就返回空结果，由 image_channels 按
  「文本为空 / 置信度低」判定落哪个码。规则集中在一处，引擎可独立替换。
"""

from dataclasses import dataclass
from typing import Protocol

from ragv1 import config


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float
    lines: tuple[str, ...] = ()


class OcrEngine(Protocol):
    def extract(self, image: bytes) -> OcrResult: ...


class RapidOcrEngine:
    """RapidOCR（PP-OCR 的 onnx 版本）：纯 pip、CPU 可跑、中英都行。"""

    def __init__(self, min_conf: float = config.OCR_MIN_CONF):
        self.min_conf = min_conf
        self._impl = None  # 惰性：首次 extract 才构造

    def _engine(self):
        if self._impl is None:
            from rapidocr_onnxruntime import RapidOCR

            self._impl = RapidOCR()
        return self._impl

    def extract(self, image: bytes) -> OcrResult:
        # ⚠️ 这里**不**吞异常。本模块的契约是「只管抽字，不管降级码」：
        # 吞掉异常会让「引擎挂了」与「图里没字」变得无法区分，`ocr_failed`
        # 也就永远打不出来。异常交给 image_channels._run_ocr 去映射成码。
        raw, _elapse = self._engine()(image)

        # 无文字时 rapidocr 返回 None 或空列表，不能直接迭代
        if not raw:
            return OcrResult("", 0.0)

        lines: list[str] = []
        scores: list[float] = []
        for item in raw:
            if not item or len(item) < 3:
                continue
            text = str(item[1]).strip()
            if not text:
                continue
            lines.append(text)
            try:
                scores.append(float(item[2]))
            except (TypeError, ValueError):
                continue

        if not lines:
            return OcrResult("", 0.0)

        confidence = sum(scores) / len(scores) if scores else 0.0
        return OcrResult("\n".join(lines), confidence, tuple(lines))


def build_ocr_engine(
    name: str, min_conf: float = config.OCR_MIN_CONF
) -> OcrEngine | None:
    """按名字构造引擎。"none" 与未知名字都返回 None（干净地降级，不抛）。"""
    if name == "rapidocr":
        return RapidOcrEngine(min_conf=min_conf)
    return None
