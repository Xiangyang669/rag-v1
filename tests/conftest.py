"""测试公共工具。

测试只验管道行为，不验 embedding 模型质量——统一用确定性假向量，
避免整套测试依赖网络与付费 API。
"""

import hashlib

DIM = 128


def fake_embed(text: str) -> list[float]:
    """确定性假向量：按字符袋构造并归一化。

    用 hashlib 而非内置 hash()——后者受 PYTHONHASHSEED 影响，跨进程不稳定。
    """
    vec = [0.0] * DIM
    for ch in text:
        h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16) % DIM
        vec[h] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec
