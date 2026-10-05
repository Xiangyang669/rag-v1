# rag-v1 检索服务镜像。
#
# 只装**服务运行期**需要的依赖：不含 pytest（测试依赖），也不含
# rapidocr-onnxruntime（图片通道的引擎，服务只做检索，不重新入库）。
#
# 索引与语料**不打包进镜像**——它们由 compose 以只读卷挂载（见 compose 注）。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN pip install --no-cache-dir \
      chromadb \
      networkx \
      jieba \
      langgraph \
      fastapi \
      uvicorn \
      openai \
      langfuse

COPY pyproject.toml ./
COPY ragv1 ./ragv1

# 容器内默认值：监听所有网卡（容器外才能访问）、索引走挂载点
ENV RAGV1_INDEX_DIR=/data/index \
    RAGV1_HOST=0.0.0.0 \
    RAGV1_PORT=8000

EXPOSE 8000

# 装配入口（ragv1/api/server.py 的 main）——不是 uvicorn 直接指 app，
# 因为 app 需要先按 RAGV1_INDEX_DIR 把三路 store 接起来
CMD ["python", "-m", "ragv1.api.server"]
