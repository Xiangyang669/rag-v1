"""压测 `/search`：P50 / P95 / P99 延迟、QPS、错误率。

面试要的几个数字，这里一次测全：

| 数字 | 怎么来 |
|---|---|
| **P95 延迟** | 每请求计时，取分位数 |
| **QPS** | 总请求数 / 总墙钟时间（并发压出来） |
| **一次查询的外部调用** | **1 次 embedding + 0 次 LLM**（见下） |
| **一次查询的 token** | `--token-probe`：直接问 embedding API 的 `usage.prompt_tokens` |

**关于"几次 LLM 调用"**：rag_v1 是**检索内核，它不调 LLM**。一次查询只
在向量路把 query 编码成向量（1 次 embedding API 往返），融合/全文/图谱
全在本机。**"几次 LLM 调用"那个问题的完整版属于回答层（cs_agent）**——
检索层给的是「**零 LLM 调用**，把大模型留给生成」，这本身就是分层取舍。

用法：

    py scripts/bench.py                                   # 打本机默认端口
    py scripts/bench.py --url http://127.0.0.1:8000/search --requests 300 --concurrency 20
    py scripts/bench.py --token-probe                     # 顺手测一次 query 的 token
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 一组真实的技术文档 query —— 刻意覆盖三路各自的强项：
# 精确术语（全文路）、口语化提问（向量路）、实体名（图谱路）
QUERIES = [
    "requests.get timeout parameter",
    "how to configure the admin service",
    "health check endpoint",
    "which file formats are supported",
    "how to set up a model provider",
    "delete user API reference",
    "what is a chunk",
    "docker compose deployment",
    "how does the pdf parser work",
    "retry policy for failed tasks",
]


def _one(
    url: str, query: str, k: int, timeout: float, paths: list[str] | None = None
) -> tuple[float, int]:
    payload: dict = {"query": query, "k": k}
    if paths is not None:
        payload["paths"] = paths
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            resp.read()
        return time.perf_counter() - started, resp.status
    except urllib.error.HTTPError as e:
        return time.perf_counter() - started, e.code


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def run_bench(
    url: str,
    requests: int,
    concurrency: int,
    k: int,
    warmup: int,
    paths: list[str] | None = None,
) -> int:
    # 预热：排除首次请求的建图/连接开销，否则 P95 会被冷启动污染
    for i in range(warmup):
        _one(url, QUERIES[i % len(QUERIES)], k, 30.0, paths)

    latencies: list[float] = []
    errors = 0

    def job(i: int):
        return _one(url, QUERIES[i % len(QUERIES)], k, 30.0, paths)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for latency, status in pool.map(job, range(requests)):
            if status != 200:
                errors += 1
            else:
                latencies.append(latency)
    wall = time.perf_counter() - started

    # 只有向量路才要算 query 的 embedding —— 关掉它就能测出"纯本地检索"的延迟
    external = "1 次 embedding（向量路）+ 0 次 LLM"
    if paths is not None and "vector" not in paths:
        external = "0 次外部调用（本地三路补齐）"

    print(f"# 目标      : {url}")
    print(f"# 通路      : {paths if paths is not None else '全部（vector+fulltext+graph）'}")
    print(f"# 请求数    : {requests}（并发 {concurrency}，预热 {warmup}）")
    print(f"# 墙钟      : {wall:.2f}s")
    print()
    print(f"QPS          : {requests / wall:.1f}")
    print(f"错误数       : {errors}")
    if latencies:
        print(f"P50 延迟     : {_percentile(latencies, 50) * 1000:.1f} ms")
        print(f"P95 延迟     : {_percentile(latencies, 95) * 1000:.1f} ms")
        print(f"P99 延迟     : {_percentile(latencies, 99) * 1000:.1f} ms")
        print(f"平均延迟     : {statistics.mean(latencies) * 1000:.1f} ms")
        print(f"最大延迟     : {max(latencies) * 1000:.1f} ms")
    print()
    print(f"一次查询的外部调用：{external}")
    return 0 if errors == 0 else 1


def token_probe() -> int:
    """问 embedding API：这组 query 的 input token 有多大。

    ⚠️ 必须显式传 `base_url=BASE_URL`——`OpenAI()` 的默认端点是
    **api.openai.com**，不是 SiliconFlow。漏了它就会去连一个连不上的地址，
    表现成"Request timed out"，很容易误判成网络问题。
    """
    from openai import OpenAI

    from ragv1.embedding import BASE_URL, EMBED_MODEL, resolve_api_key

    client = OpenAI(api_key=resolve_api_key(), base_url=BASE_URL, timeout=60.0)
    resp = client.embeddings.create(model=EMBED_MODEL, input=QUERIES)
    total = resp.usage.prompt_tokens
    print(f"query 条数        : {len(QUERIES)}")
    print(f"总 token          : {total}")
    print(f"平均每条 query    : {total / len(QUERIES):.1f} token")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="压测 /search")
    ap.add_argument("--url", default="http://127.0.0.1:8000/search")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--paths",
        default=None,
        help="限定通路，逗号分隔（vector,fulltext,graph）。"
        "传 fulltext,graph 可测**纯本地检索延迟**（不调 embedding API）",
    )
    ap.add_argument("--token-probe", action="store_true", help="只测 query 的 token 量")
    args = ap.parse_args()

    if args.token_probe:
        return token_probe()

    paths = args.paths.split(",") if args.paths else None
    return run_bench(
        args.url, args.requests, args.concurrency, args.k, args.warmup, paths
    )


if __name__ == "__main__":
    raise SystemExit(main())
