"""生产装配的端到端：**落盘的**三路索引 → 一个能服务的 app。

这是「部署」这块的守卫：每个 store 自己的单测都绿，不等于它们被**接起来**
还能服务——`build_app` 之前根本不存在，`create_app` 只在测试里被注入过。

⚠️ 注入 `fake_embed`（确定性假向量），不发真 API 调用。
"""

from pathlib import Path

import pytest
from conftest import fake_embed
from fastapi.testclient import TestClient

from ragv1.api.server import _chunks_by_id, build_app
from ragv1.ingest.build import build_corpus
from ragv1.store.fts_store import FtsStore

FIXTURES = Path(__file__).parent / "fixtures"


def _build(tmp_path, name="kb"):
    assert build_corpus(FIXTURES, tmp_path / name, embed_fn=fake_embed) > 0
    return tmp_path / name


def test_build_app_serves_from_persisted_index(tmp_path):
    """从落盘索引装配出来的 app，真的能检索（不是空结果）。"""
    idx = _build(tmp_path)
    client = TestClient(build_app(idx, embed_fn=fake_embed))

    resp = client.post("/search", json={"query": "检索参数 k 默认值", "k": 5})
    assert resp.status_code == 200

    body = resp.json()
    assert body["results"], "装配后的服务检索为空——三路没接上"
    assert "degraded" in body
    # 元数据从回源表查出来注入（doc_id 来自 meta_lookup）
    assert body["results"][0]["doc_id"]


def test_build_app_filters_paths(tmp_path):
    """按 paths 限路仍生效——装配没把开关丢掉。"""
    idx = _build(tmp_path)
    client = TestClient(build_app(idx, embed_fn=fake_embed))

    resp = client.post(
        "/search", json={"query": "k", "k": 5, "paths": ["fulltext"]}
    )
    assert resp.status_code == 200
    for item in resp.json()["results"]:
        assert item["sources"] == ["fulltext"]


def test_build_app_missing_index_exits(tmp_path):
    """索引目录不存在时，报错要指名道姓地告诉你怎么建。"""
    with pytest.raises(SystemExit, match="build_index"):
        build_app(tmp_path / "not-built")


def test_chunks_by_id_covers_every_chunk(tmp_path):
    """图谱路的 membership 集合必须覆盖回源表里的每一个块。

    少一个块 = 图谱路对它永远视而不见，而每个模块自己的测试仍是绿的
    ——正是那种「单测全绿、接起来才炸」的集成缺口。
    """
    idx = _build(tmp_path)
    fts = FtsStore(idx / "kb.db")
    assert set(_chunks_by_id(fts)) == fts.chunk_ids()
