"""全文索引（SQLite FTS5 + 内置 bm25）。

两个表：`chunks` 存**原文 + 来源元数据**（供回源），`chunks_fts` 是检索用的
倒排表，存的是**分词后**的文本。两者用 chunk_id 关联——chunk_id 由 ingest 层
决定性生成，与向量路同一套。

⚠️ 分词只作用于 `chunks_fts`。`chunks.text` 必须保持原文，否则回源的
块内容会变成空格分隔的词串。见 tests/test_chinese.py。

⚠️ `bm25()` 返回的是**负值，越小越相关**。按它升序排即最相关在前。

⚠️ 列名 `order` 是 SQL 关键字，读写两处的 SQL 都必须加双引号。
"""

import json
import sqlite3
from pathlib import Path

from ragv1.text.tokenize import tokenize_for_index
from ragv1.types import Chunk, Hit

# 元数据 schema 版本。改动 chunks 表的列结构时必须 +1。
SCHEMA_VERSION = 2

_META_COLUMNS = (
    "doc_id",
    "kind",
    "page",
    '"order"',
    "part",
    "bbox",
    "image_ref",
    "image_path",
    "table_structured",
    "degrade",
)


def _match_expr(query: str) -> str:
    """把查询转成安全的 FTS5 MATCH 表达式。

    FTS5 的 MATCH 有自己的一套语法（AND/OR/NEAR/引号/星号…），
    直接把裸查询丢进去会被当语法解析。这里逐个 token 加引号，
    空白连接即隐式 AND。
    """
    tokens = [t.replace('"', '""') for t in query.split() if t.strip()]
    return " ".join(f'"{t}"' for t in tokens)


class FtsStore:
    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._con = sqlite3.connect(self._db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """建立/升级表结构。

        版本不符时**整表重建**：索引是派生的、永远全量重建的数据，保留旧结构
        收益为零，而按旧结构读新列会静默拿错数据——重建是最安全的选择。
        """
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL)"
        )
        row = self._con.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        current = int(row[0]) if row else None

        if current != SCHEMA_VERSION:
            self._con.execute("DROP TABLE IF EXISTS chunks")
            self._con.execute("DROP TABLE IF EXISTS chunks_fts")
            self._con.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

        self._con.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "  chunk_id TEXT PRIMARY KEY,"
            "  doc_id TEXT NOT NULL,"
            "  heading_path TEXT NOT NULL,"
            "  text TEXT NOT NULL,"
            "  kind TEXT NOT NULL DEFAULT 'text',"
            "  page INTEGER,"
            '  "order" INTEGER NOT NULL DEFAULT 0,'
            "  part INTEGER NOT NULL DEFAULT 0,"
            "  bbox TEXT,"
            "  image_ref TEXT,"
            "  image_path TEXT,"
            "  table_structured INTEGER NOT NULL DEFAULT 1,"
            "  degrade TEXT NOT NULL DEFAULT '[]')"
        )
        self._con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "  chunk_id UNINDEXED,"
            "  text,"
            "  tokenize='unicode61')"
        )
        self._con.commit()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [(c.chunk_id,) for c in chunks]

        self._con.executemany(
            "INSERT OR REPLACE INTO chunks"
            " (chunk_id, doc_id, heading_path, text, kind, page, \"order\", part,"
            "  bbox, image_ref, image_path, table_structured, degrade)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    "/".join(c.heading_path),
                    c.text,
                    c.kind,
                    c.page,
                    c.order,
                    c.part,
                    json.dumps(list(c.bbox)) if c.bbox is not None else None,
                    c.image_ref,
                    c.image_path,
                    int(c.table_structured),
                    json.dumps(list(c.degrade)),
                )
                for c in chunks
            ],
        )
        # ⚠️ fts5 表没有唯一约束，裸 INSERT 会让重跑入库产生重复行 ——
        # 重复的块在 RRF 里会拿到 2/(k+rank)，静默污染评估数字。
        # 先按 chunk_id 清掉旧行，与 VectorStore / GraphStore 的幂等性对齐。
        self._con.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", ids)
        # 只有这一张表存分词结果；上面那张表永远是原文
        self._con.executemany(
            "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
            [(c.chunk_id, tokenize_for_index(c.text)) for c in chunks],
        )
        self._con.commit()

    def search(self, query: str, k: int) -> list[Hit]:
        if not query.strip():
            return []
        expr = _match_expr(tokenize_for_index(query))
        if not expr:
            return []
        rows = self._con.execute(
            "SELECT chunk_id, bm25(chunks_fts) AS score"
            " FROM chunks_fts WHERE chunks_fts MATCH ?"
            " ORDER BY score LIMIT ?",
            (expr, k),
        ).fetchall()
        return [
            Hit(chunk_id=cid, rank=i + 1, score=score, path="fulltext")
            for i, (cid, score) in enumerate(rows)
        ]

    def chunk_ids(self) -> set[str]:
        return {row[0] for row in self._con.execute("SELECT chunk_id FROM chunks")}

    def text_of(self, chunk_id: str) -> str | None:
        """回源取块原文（不是分词后的检索文本）。"""
        row = self._con.execute(
            "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return row[0] if row else None

    def meta_of(self, chunk_id: str) -> dict | None:
        """回源取块的来源元数据。块不存在返回 None。

        bbox / degrade 以 JSON 文本存储，这里反序列化回 Python 值——
        调用方拿到的是可直接用的形状（列表），不用关心存储细节。
        """
        row = self._con.execute(
            f"SELECT {', '.join(_META_COLUMNS)} FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return None

        (
            doc_id,
            kind,
            page,
            order,
            part,
            bbox,
            image_ref,
            image_path,
            table_structured,
            degrade,
        ) = row
        return {
            "doc_id": doc_id,
            "kind": kind,
            "page": page,
            "order": order,
            "part": part,
            "bbox": json.loads(bbox) if bbox is not None else None,
            "image_ref": image_ref,
            "image_path": image_path,
            "table_structured": bool(table_structured),
            "degrade": json.loads(degrade) if degrade is not None else [],
        }
