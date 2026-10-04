"""多模态通道：理解图片的**视觉内容**（流程图/图表/截图/照片）。

OCR 拿得到图上的字，拿不到那层字表达不出的东西：流程的走向、图表的趋势、
模块之间的层级。这些正是多模态通道要产出的。

设计要点：

- **复用本项目的 SiliconFlow 凭据**（与 embedding 同一个 key 与 base_url），
  不另造一套配置。
- **结构化输出**：提示词要求只回 JSON，四个字段固定。解析**宽松但保守**——
  能剥围栏、能从客套话里截出 JSON；实在解析不出来时把原始返回当描述留下，
  绝不让一张图变成空。
- **没有凭据就返回 None**：由调用方落 `vlm_unavailable`，而不是启动时崩。
"""

import base64
import json
import re
from dataclasses import dataclass
from typing import Protocol

from ragv1 import config
from ragv1.embedding import BASE_URL, resolve_api_key

VISION_PROMPT = """你是文档理解助手。仔细看这张图片，只输出一个 JSON 对象，不要输出任何其它文字。
字段固定为以下四个，全部用中文作答：

{
  "image_type": "流程图 | 图表 | 截图 | 照片 | 其他",
  "content": "图片的核心内容，一到三句话",
  "data": "图中的关键数据、数值、结论；没有就填空字符串",
  "relations": "元素之间的关系（流程走向、层级、对比）；没有就填空字符串"
}"""

# ```json ... ``` 或 ``` ... ``` 围栏
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass(frozen=True)
class VisionResult:
    image_type: str
    content: str
    data: str = ""
    relations: str = ""
    raw: str = ""  # 原始返回文本，解析失败时它就是唯一的内容来源


class VisionEngine(Protocol):
    def describe(self, image: bytes) -> VisionResult: ...


def _extract_json_text(raw: str) -> str | None:
    """从一段可能带围栏/客套话的文本里截出 JSON 子串。

    先剥围栏（VLM 常见行为），再取最外层的一对花括号。**不用贪婪正则找
    JSON** —— 嵌套花括号下它会截错。
    """
    match = _FENCE_RE.search(raw)
    text = match.group(1) if match else raw
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def parse_vision_json(raw: str) -> VisionResult | None:
    """宽松解析。解析不出来返回 None（调用方据此保留原文并落码）。"""
    candidate = _extract_json_text(raw or "")
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    def field(name: str) -> str:
        value = obj.get(name, "")
        return "" if value is None else str(value)

    return VisionResult(
        image_type=field("image_type"),
        content=field("content"),
        data=field("data"),
        relations=field("relations"),
        raw=raw,
    )


class SiliconFlowVision:
    """走 SiliconFlow 的 OpenAI 兼容接口调视觉模型。"""

    def __init__(self, model: str = config.VLM_MODEL, api_key: str | None = None, client=None):
        self.model = model
        self._api_key = api_key
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key or resolve_api_key(), base_url=BASE_URL
            )
        return self._client

    def describe(self, image: bytes) -> VisionResult:
        data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = parse_vision_json(raw)
        if parsed is not None:
            return parsed

        # 解析失败：把原文当描述留下。宁可描述粗糙，也不要一张图变空白。
        return VisionResult(image_type="其他", content=raw[:200], raw=raw)


def _has_credentials() -> bool:
    try:
        return bool(resolve_api_key())
    except RuntimeError:
        return False


def build_vision_engine(name: str) -> VisionEngine | None:
    """按名字构造引擎。未知名字或没有凭据都返回 None（干净降级）。"""
    if name != "siliconflow":
        return None
    if not _has_credentials():
        return None
    return SiliconFlowVision()
