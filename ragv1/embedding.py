"""Embedding 客户端。

走 SiliconFlow 的 OpenAI 兼容接口调 BAAI/bge-m3——不在本地加载模型，
因此不需要 sentence_transformers。

**凭据复用 practice/.env**（不再重造一套配置）：`load_key` 的写法与用法
沿用 `practice/m3_kb.py` 里的同名函数，只是不再 SystemExit——库代码不该
直接杀进程。
"""

import os
from pathlib import Path
from typing import Callable

EMBED_MODEL = "BAAI/bge-m3"
BASE_URL = "https://api.siliconflow.cn/v1"
KEY_NAME = "SILICONFLOW_API_KEY"

# rag_v1/ragv1/embedding.py → rag_v1/ → 上层目录
_PROJECT_DIR = Path(__file__).resolve().parents[1]
_REPO_DIR = _PROJECT_DIR.parent

# 凭据文件候选，**就近优先**：
#   1. 项目自己的 .env —— rag_v1 被单独抽成仓库时（对外发的版本）就放这里
#   2. 上层 practice/.env —— 留在 ai-agent-90days 里时的既有布局
# 两种布局都能跑，抽出/嵌入都不用改代码。
ENV_FILE_CANDIDATES: tuple[Path, ...] = (
    (_PROJECT_DIR / ".env").resolve(),
    (_REPO_DIR / "practice" / ".env").resolve(),
)

# 兼容旧引用：默认候选文件
DEFAULT_ENV_FILE = ENV_FILE_CANDIDATES[0]


def load_key(name: str, path: str | Path) -> str:
    """从 .env 风格的文本文件里读一个键（写法沿用 practice/m3_kb.py）。"""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as err:
        raise RuntimeError(f"没找到 {name}：凭据文件不可读（{path}）") from err

    for line in text.splitlines():
        line = line.strip()
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"没找到 {name}（在 {path} 中）")


def resolve_api_key(env_file: str | Path | None = None) -> str:
    """凭据解析顺序：环境变量 → 调用方指定的文件 → 依次尝试 ENV_FILE_CANDIDATES。

    不得硬编码（见计划的 Global Constraints）。

    ⚠️ 早先的版本是「从当前目录向上找 .env」，但 practice/ 不是 rag_v1/ 的
    祖先——生产路径永远找不到 key。改成显式定位，见 tests/test_embedding.py。
    """
    from_env = os.environ.get(KEY_NAME)
    if from_env:
        return from_env

    if env_file is not None:
        return load_key(KEY_NAME, env_file)

    for candidate in ENV_FILE_CANDIDATES:
        if candidate.is_file():
            return load_key(KEY_NAME, candidate)

    tried = " 或 ".join(str(c) for c in ENV_FILE_CANDIDATES)
    raise RuntimeError(f"没找到 {KEY_NAME}：请设置该环境变量，或在 {tried} 放置 .env 文件")


def default_embed_fn() -> Callable[[str], list[float]]:
    """构造真实 embedder。只在生产装配路径调用，测试请注入假向量。"""
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(), base_url=BASE_URL)

    def embed(text: str) -> list[float]:
        resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
        return resp.data[0].embedding

    return embed
