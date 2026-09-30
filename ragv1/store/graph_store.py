"""图谱存储：SQLite 持久化 + NetworkX 内存遍历。

零依赖栈下**不用图数据库**——三张表足够：
- `nodes`          实体
- `edges`          关系
- `entity_chunks`  **posting list**：实体出现在哪些块

posting list 是硬约束二：没有它，图谱路抽完实体回不到原文，整条路作废。
"""

import sqlite3
from pathlib import Path

import networkx as nx

from ragv1.ingest.extract import entities_of_chunk, extract
from ragv1.types import Chunk

ALIAS_SEP = ","


class GraphStore:
    def __init__(self, db_path: str | Path):
        self._con = sqlite3.connect(str(db_path))
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS nodes ("
            "  canonical TEXT PRIMARY KEY,"
            "  kind TEXT NOT NULL,"
            "  aliases TEXT NOT NULL)"
        )
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS edges ("
            "  src TEXT NOT NULL, dst TEXT NOT NULL, kind TEXT NOT NULL)"
        )
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS entity_chunks ("
            "  entity TEXT NOT NULL, chunk_id TEXT NOT NULL)"
        )
        self._con.commit()
        self._g = nx.DiGraph()

    def build(self, chunks: list[Chunk]) -> None:
        """从同一套块建图。幂等：重复 build 不会累积重复行。"""
        entities, relations = extract(chunks)

        self._con.execute("DELETE FROM nodes")
        self._con.execute("DELETE FROM edges")
        self._con.execute("DELETE FROM entity_chunks")

        self._con.executemany(
            "INSERT OR REPLACE INTO nodes (canonical, kind, aliases) VALUES (?, ?, ?)",
            [(e.canonical, e.kind, ALIAS_SEP.join(e.aliases)) for e in entities],
        )
        self._con.executemany(
            "INSERT INTO edges (src, dst, kind) VALUES (?, ?, ?)",
            [(r.src, r.dst, r.kind) for r in relations],
        )
        self._con.executemany(
            "INSERT INTO entity_chunks (entity, chunk_id) VALUES (?, ?)",
            [(name, c.chunk_id) for c in chunks for name in entities_of_chunk(c)],
        )
        self._con.commit()

        self._g = nx.DiGraph()
        for e in entities:
            self._g.add_node(e.canonical, kind=e.kind)
        for r in relations:
            self._g.add_edge(r.src, r.dst, kind=r.kind)

    def params_of(self, entity: str) -> list[str]:
        if entity not in self._g:
            return []
        return sorted(
            n
            for n in self._g.successors(entity)
            if self._g.nodes[n].get("kind") == "parameter"
        )

    def chunks_of(self, entity: str) -> list[str]:
        """posting list：该实体出现在哪些块。"""
        return sorted(
            row[0]
            for row in self._con.execute(
                "SELECT chunk_id FROM entity_chunks WHERE entity = ?", (entity,)
            )
        )

    def neighbors(self, entity: str) -> list[str]:
        """一跳邻居（前驱 + 后继），已排序以保证可复现。

        图谱路检索器靠它做「用户没提、但相关」的那一跳扩张。
        """
        if entity not in self._g:
            return []
        return sorted(set(self._g.successors(entity)) | set(self._g.predecessors(entity)))

    def aliases(self) -> dict[str, str]:
        """别名 → canonical。canonical 自身也映射到自己。"""
        out: dict[str, str] = {}
        for canonical, aliases in self._con.execute(
            "SELECT canonical, aliases FROM nodes ORDER BY canonical"
        ):
            out.setdefault(canonical, canonical)
            for alias in (aliases or "").split(ALIAS_SEP):
                alias = alias.strip()
                if alias:
                    out.setdefault(alias, canonical)
        return out
