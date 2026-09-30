"""可观测：Langfuse 埋点。

**没配置 key 时返回 no-op**——本地开发与 CI 不该因为缺少可观测平台就跑不起来，
也绝不允许因为没有 key 而抛错。

⚠️ 这一处直接复用了 `practice/m5_langfuse_demo.py` 里记下的坑：
   **Langfuse SDK 读的环境变量是 `LANGFUSE_HOST`，而不是 `.env` 里登记的
   `LANGFUSE_BASE_URL`。** 名字不对时 SDK **不报错**，只是默默连到默认的
   `cloud.langfuse.com` —— trace 就发到别人家去了。
   所以这里必须显式转一手（见 `tracer()`）。
"""

import os
from contextlib import contextmanager

from ragv1.embedding import DEFAULT_ENV_FILE, load_key

PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY = "LANGFUSE_SECRET_KEY"
BASE_URL = "LANGFUSE_BASE_URL"  # .env 里登记的名字
HOST_ENV = "LANGFUSE_HOST"  # SDK 实际读的名字
DEFAULT_HOST = "https://cloud.langfuse.com"


def _setting(name: str) -> str | None:
    """环境变量优先，其次 practice/.env。取不到返回 None（不抛错）。"""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return load_key(name, DEFAULT_ENV_FILE)
    except RuntimeError:
        return None


def tracer():
    """构造 Langfuse 客户端；缺 key 时返回 None，调用方按 no-op 走。"""
    public_key = _setting(PUBLIC_KEY)
    secret_key = _setting(SECRET_KEY)
    if not (public_key and secret_key):
        return None

    # 关键一行：把 .env 里的 BASE_URL 转成 SDK 认的 HOST
    os.environ[HOST_ENV] = _setting(BASE_URL) or DEFAULT_HOST

    from langfuse import get_client

    return get_client()


@contextmanager
def span(tracer, name: str, as_type: str = "retriever"):
    """一层 span。名字用调用方给的语义名——本项目里就是路径名。

    tracer 为 None 时是 no-op：整个 with 块照常执行，不产生任何外部调用。

    ⚠️ langfuse v4 的方法名是 `start_as_current_observation`，**不是**
    `start_as_current_span`（v2/v3 时代叫后者）。写错时无 key 的测试路径
    全程 no-op、看不出问题，只有真接了 Langfuse 才会崩
    —— 所以 tests/test_tracing.py 里专门拿一个「只实现真实 API」的假客户端钉住它。
    """
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_observation(name=name, as_type=as_type) as current:
        yield current
