"""FastAPI 检索端点。

V1 **不加鉴权**（spec 的 Out-of-scope）。

关键设计：`create_app` 接的是 **retriever**，不是预建好的图。因为三路开关
是在**构建期**决定节点存不存在的（T9），预建图没法按请求变化——只能事后
过滤结果，而那会错误地丢掉被「已启用路 + 已禁用路」同时命中的块。
"""

import pytest
from fastapi.testclient import TestClient

from ragv1.types import Hit
from ragv1.retrieve import ALL_PATHS, Retriever
from ragv1.api.main import create_app

# 路径名 → Retriever 构造参数名（全文路路径名 fulltext、属性名 fts）
_KWARG = {"vector": "vector", "fulltext": "fts", "graph": "graph"}


class Ok:
    def __init__(self, name):
        self.name = name

    def retrieve(self, query, k):
        return [Hit(self.name + "-c", 1, 1.0, self.name)]


class Boom:
    def retrieve(self, query, k):
        raise RuntimeError("炸了")


def make_retriever():
    return Retriever(**{_KWARG[p]: Ok(p) for p in ALL_PATHS})


@pytest.fixture
def client():
    return TestClient(create_app(make_retriever()))


def test_search_returns_structured_results(client):
    r = client.post("/search", json={"query": "q", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["chunk_id"]
    assert set(body["results"][0]["sources"])
    assert body["results"][0]["rank"] == 1


def test_paths_param_limits_paths(client):
    r = client.post("/search", json={"query": "q", "k": 5, "paths": ["graph"]})
    assert {s for x in r.json()["results"] for s in x["sources"]} == {"graph"}


def test_blank_query_returns_4xx_not_500(client):
    r = client.post("/search", json={"query": "   ", "k": 5})
    assert 400 <= r.status_code < 500


def test_default_k_is_10(client):
    r = client.post("/search", json={"query": "q"})
    assert r.status_code == 200


def test_degradation_is_reported_in_response():
    retriever = Retriever(
        vector=Ok("vector"), fts=Ok("fulltext"), graph=Boom()
    )
    c = TestClient(create_app(retriever))
    body = c.post("/search", json={"query": "q", "k": 5}).json()
    assert "graph" in body["degraded"]
    assert body["results"], "有一路挂了，另外两路的结果仍应返回"


def test_empty_paths_returns_empty_results(client):
    r = client.post("/search", json={"query": "q", "k": 5, "paths": []})
    assert r.status_code == 200
    assert r.json()["results"] == []


# ── 来源元数据的端到端暴露（Task 9）────────────────────────────
# 融合层不认识 store（rrf_fuse 的契约是「只做排名」），所以查表在 API 边界
# 注入：缺省 None 时行为与改造前完全一致。


def test_metadata_is_absent_when_lookup_not_supplied():
    """缺省 None 时响应体里连字段都不该出现（不是 null）"""
    app = create_app(make_retriever())
    r = TestClient(app).post("/search", json={"query": "q", "k": 3})
    assert r.status_code == 200
    assert "doc_id" not in r.json()["results"][0]


def test_metadata_is_returned_when_lookup_supplied():
    def lookup(chunk_id):
        return {
            "doc_id": "d.md", "kind": "table", "page": 3, "order": 7,
            "part": 1, "bbox": [1.0, 2.0, 3.0, 4.0],
            "image_ref": "http://x/y.png", "image_path": None,
            "table_structured": False, "degrade": ["table_unstructured"],
        }

    app = create_app(make_retriever(), meta_lookup=lookup)
    item = TestClient(app).post("/search", json={"query": "q", "k": 3}).json()["results"][0]
    assert item["doc_id"] == "d.md" and item["kind"] == "table"
    assert item["page"] == 3 and item["order"] == 7 and item["part"] == 1
    assert item["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert item["degrade"] == ["table_unstructured"]
    assert item["table_structured"] is False


def test_lookup_returning_none_does_not_crash():
    """chunk 不在回源表里（例如索引与元数据不同步）时不能 500"""
    app = create_app(make_retriever(), meta_lookup=lambda cid: None)
    r = TestClient(app).post("/search", json={"query": "q", "k": 3})
    assert r.status_code == 200
