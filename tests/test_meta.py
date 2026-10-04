"""来源元数据的落地与回源。

FTS 的 chunks 表是项目里唯一的**回源表**（text_of 从它取原文），所以元数据
也放这里：一次查询就能拿到「这段文字出自哪篇文档、哪一页、是不是表格」。

schema 版本守卫是有意的破坏性行为：版本不符就整表重建。索引是派生的、
永远全量重建的数据，保留旧结构的收益为零，而误读旧结构的代价是拿错列。
"""

import json

from ragv1.store.fts_store import FtsStore
from ragv1.store.vector_store import chunk_metadata
from ragv1.types import Chunk


def _chunk(**kw):
    base = dict(chunk_id="c1", doc_id="d.md", heading_path=("H",), text="正文")
    base.update(kw)
    return Chunk(**base)


def test_meta_of_returns_all_keys(tmp_path):
    store = FtsStore(tmp_path / "kb.db")
    store.add([
        _chunk(
            kind="table", page=3, order=7, part=1,
            bbox=(1.0, 2.0, 3.0, 4.0),
            image_ref="http://x/y.png", image_path="/tmp/y.png",
            table_structured=False, degrade=("table_unstructured",),
        )
    ])
    meta = store.meta_of("c1")
    assert set(meta) == {
        "doc_id", "kind", "page", "order", "part", "bbox", "image_ref",
        "image_path", "table_structured", "degrade",
    }
    assert meta["doc_id"] == "d.md"
    assert meta["kind"] == "table"
    assert meta["page"] == 3 and meta["order"] == 7 and meta["part"] == 1
    assert meta["bbox"] == [1.0, 2.0, 3.0, 4.0]  # 已反序列化
    assert meta["degrade"] == ["table_unstructured"]  # 已反序列化
    assert meta["table_structured"] is False


def test_meta_of_defaults_for_plain_text(tmp_path):
    store = FtsStore(tmp_path / "kb.db")
    store.add([_chunk()])
    meta = store.meta_of("c1")
    assert meta["kind"] == "text" and meta["page"] is None
    assert meta["bbox"] is None and meta["degrade"] == []
    assert meta["table_structured"] is True


def test_meta_of_unknown_chunk_returns_none(tmp_path):
    assert FtsStore(tmp_path / "kb.db").meta_of("nope") is None


def test_old_schema_is_dropped_and_recreated(tmp_path):
    """版本不符时整表重建——索引是派生数据，全量重建无损失"""
    db = tmp_path / "kb.db"
    FtsStore(db).add([_chunk()])
    # 伪造旧版本
    import sqlite3

    con = sqlite3.connect(db)
    con.execute("UPDATE schema_meta SET value = '1' WHERE key = 'schema_version'")
    con.commit()
    con.close()

    assert FtsStore(db).meta_of("c1") is None  # 旧数据被清掉
    assert FtsStore(db).chunk_ids() == set()


def test_repeated_add_does_not_duplicate_fts_rows(tmp_path):
    """重跑入库不能产生重复行 —— 重复块会让 RRF 给同一块加两次分"""
    store = FtsStore(tmp_path / "kb.db")
    store.add([_chunk()])
    store.add([_chunk()])
    assert len(store.search("正文", k=10)) == 1


# ── 向量路的元数据（Task 7）────────────────────────────────────


def test_chunk_metadata_only_uses_chroma_scalar_types():
    """Chroma 的 metadata 只接受 str/int/float/bool —— 元组和 None 都不行"""
    meta = chunk_metadata(
        _chunk(
            kind="image", page=2, order=5, part=0,
            bbox=(1.0, 2.0, 3.0, 4.0),
            image_ref="http://x/y.png", image_path=None,
            table_structured=True, degrade=("ocr_empty",),
        )
    )
    for k, v in meta.items():
        assert isinstance(v, (str, int, float, bool)), (
            f"{k} 的类型 {type(v)} 不被 Chroma 接受"
        )
    assert meta["degrade"] == json.dumps(["ocr_empty"])  # 元组 → JSON 字符串
    assert meta["bbox"] == json.dumps([1.0, 2.0, 3.0, 4.0])
    assert meta["kind"] == "image" and meta["page"] == 2
    assert "image_path" not in meta  # None 的键直接省略


def test_chunk_metadata_omits_all_none():
    meta = chunk_metadata(_chunk())
    assert meta == {
        "doc_id": "d.md",
        "kind": "text",
        "order": 0,
        "part": 0,
        "table_structured": True,
        "degrade": "[]",
    }
