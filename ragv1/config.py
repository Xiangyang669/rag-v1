"""路径与常量。

V1 的语料目录是硬编码的本机路径（见计划的 Self-Review「未决」一节）——
语料换位置时改这里即可。
"""

import os
from pathlib import Path

# V1 基准语料：结构化技术类文档（英文为主）
CORPUS_DIR = Path("C:/Users/lenovo/OneDrive/Desktop/ragflow-main/docs")

# 索引落盘目录：项目根下的 .indexes/（已在 .gitignore 中忽略）
INDEX_DIR = Path(__file__).resolve().parent.parent / ".indexes"

# ── V2 中文主语料 ────────────────────────────────────────────
# 与英文基准**分属两套索引，互不覆盖**：V1 的回归基线依赖 .indexes/kb，
# 中文索引必须另立目录，否则重建中文语料会把基线冲掉。
# 语料来源：FastGPT 文档仓库 document/content（中文 .mdx 已转为 .md）。
# 换语料位置时改这里，或用 RAGV1_ZH_CORPUS 环境变量覆盖。
ZH_CORPUS_DIR = Path(os.environ.get("RAGV1_ZH_CORPUS", "D:/corpus/fastgpt-zh"))
ZH_INDEX_DIR = INDEX_DIR / "kb_zh"

# 单块长度上限（字符数）。超过此值的节需要 OVER_CAP 再切。
MAX_CHARS = 1200

# RRF 的 k 常数（倒数排名融合）
RRF_K = 60

# ── V2 查询自适应权重（任务 6）──────────────────────────────
# 静态融合权重初值：调研查到的企业三信号示例，**必须在 dev 集上重调**（任务 8）。
# 键即「路名」，与 rrf_fuse 的 weights 同一套语义（缺失按 1.0，≤ 0 视为不参与）。
STATIC_FUSION_WEIGHTS = {"vector": 0.50, "fulltext": 0.35, "graph": 0.15}

# 改写变体的折扣（双路检索时压低改写路，避免它带偏）。
REWRITE_WEIGHT_DISCOUNT = 0.5

# 词面信号触发时，全文路的上调倍数 / 向量路的下调倍数。
LEXICAL_FULLTEXT_BOOST = 1.8
LEXICAL_VECTOR_DAMP = 0.6

# 全文路「压倒性命中」的判据：median(scores) - top1 超过此值即触发。
# ⚠️ bm25 是负值、越小越相关：故「top-1 明显强于其余」= median - top1 是一个大的**正数**。
FULLTEXT_DOMINANCE_GAP = 3.0

# 改写变体的路径名前缀/后缀标记（双路检索时改写路叫 "vector#rw" 等）。
REWRITTEN_SUFFIX = "#rw"

# ── 图片双通道 ──────────────────────────────────────────────
# 引擎名都可从环境变量覆盖：换 OCR / 换多模态模型不该改代码。
# 取值 "none" 表示显式禁用该通道。

# OCR 引擎：rapidocr（本机 CPU 可跑，纯 pip）/ none
OCR_ENGINE = os.environ.get("OCR_ENGINE", "rapidocr")

# 多模态引擎：siliconflow（复用本项目的 SILICONFLOW_API_KEY）/ none
VLM_ENGINE = os.environ.get("VLM_ENGINE", "siliconflow")

# 多模态模型名。改它不影响其它通道。
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")

# 「OCR 有料」的判据：抽到的字符数与平均置信度都要过线，
# 才认为这张图是「文字为主」，不必再花一次多模态调用。
OCR_MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", "10"))
OCR_MIN_CONF = float(os.environ.get("OCR_MIN_CONF", "0.5"))

# 图片通道总开关。关掉后图片行按普通正文处理（与阶段一行为一致）。
ENABLE_IMAGE = os.environ.get("ENABLE_IMAGE", "1") not in {"0", "false", "no"}

# 远程图片下载后的本地缓存目录（在已 gitignore 的 .indexes/ 下）
IMAGE_CACHE_DIR = INDEX_DIR / "images"
