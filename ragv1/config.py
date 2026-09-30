"""路径与常量。

V1 的语料目录是硬编码的本机路径（见计划的 Self-Review「未决」一节）——
语料换位置时改这里即可。
"""

from pathlib import Path

# V1 基准语料：结构化技术类文档（英文为主）
CORPUS_DIR = Path("C:/Users/lenovo/OneDrive/Desktop/ragflow-main/docs")

# 索引落盘目录：项目根下的 .indexes/（已在 .gitignore 中忽略）
INDEX_DIR = Path(__file__).resolve().parent.parent / ".indexes"

# 单块长度上限（字符数）。超过此值的节需要 OVER_CAP 再切。
MAX_CHARS = 1200

# RRF 的 k 常数（倒数排名融合）
RRF_K = 60
