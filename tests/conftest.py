"""测试公共工具。

测试只验管道行为，不验 embedding 模型质量——统一用确定性假向量，
避免整套测试依赖网络与付费 API。

⚠️ 契约与真实 embedder 一致：**收文本列表、返向量列表**（批量）。
"""

import hashlib

DIM = 128


def _one(text: str) -> list[float]:
    """单个文本的确定性假向量：按字符袋构造并归一化。

    用 hashlib 而非内置 hash()——后者受 PYTHONHASHSEED 影响，跨进程不稳定。
    """
    vec = [0.0] * DIM
    for ch in text:
        h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16) % DIM
        vec[h] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


def fake_embed(texts: list[str]) -> list[list[float]]:
    """批量假向量。"""
    return [_one(t) for t in texts]
