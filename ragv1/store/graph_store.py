"""图谱存储：SQLite 持久化 + NetworkX 内存遍历。

零依赖栈下**不用图数据库**——三张表足够：
- `nodes`          实体
- `edges`          关系
- `entity_chunks`  **posting list**：实体出现在哪些块

posting list 是硬约束二：没有它，图谱路抽完实体回不到原文，整条路作废。
"""

import sqlite3
import threading
from pathlib import Path

import networkx as nx

from ragv1.ingest.extract import entities_of_chunk, extract
from ragv1.types import Chunk

ALIAS_SEP = ","


class GraphStore:
    def __init__(self, db_path: str | Path):
        # ⚠️ 检索时 LangGraph 会把三路分发到**不同线程**并行执行。sqlite3
        # 默认 check_same_thread=True，只允许创建连接的那个线程使用它——于是
        # 并行调度下图谱路会抛 ProgrammingError，被降级机制静默吞成空结果。
        # 这个缺陷在假检索器的单测里看不见，只有三路真检索器跑完整编排才暴露。
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        # 单连接被多线程并发读，用一把锁串行化（读都很短）
        self._lock = threading.Lock()
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
        # ⚠️ 构造即加载：生产/评估装配路径（`api/server.py::_open`）只构造
        # GraphStore、**从不调 build()**，所以「把边读进内存」的唯一入口若只
        # 在 build() 里，检索时 `_g` 就恒为空、一跳扩张（图谱路的独家价值）
        # 端到端失效。这里在构造时就把已落盘的图读进来。
        self._load_from_db()

    def _load_from_db(self) -> None:
        """从 `nodes` / `edges` 两张表重建内存图 `_g`。

        这是把持久化图读进内存的**唯一入口**——`__init__` 与 `build()` 都调它，
        保证「新建（构造即加载）」与「build 之后」两条路径用**同一套**节点/边
        构造规则，行为不会随「这个实例是新建的还是 build 过的」而分叉。
        检索期 `neighbors()` / `params_of()` 只读这张内存图，不再查库（否则一次
        遍历会退化成 N 次 SQL）。
        """
        g = nx.DiGraph()
        with self._lock:
            node_rows = self._con.execute(
                "SELECT canonical, kind FROM nodes"
            ).fetchall()
            edge_rows = self._con.execute("SELECT src, dst, kind FROM edges").fetchall()
        for canonical, kind in node_rows:
            g.add_node(canonical, kind=kind)
        for src, dst, kind in edge_rows:
            g.add_edge(src, dst, kind=kind)
        self._g = g

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

        # 落盘后**从库重建**内存图——与 `__init__` 走同一条 `_load_from_db()`，
        # 两条路径不分叉。`build([])` 清空了三张表，重建后 `_g` 自然为空
        # （不会留下构造时载入的旧状态）。
        self._load_from_db()

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
        with self._lock:
            rows = self._con.execute(
                "SELECT chunk_id FROM entity_chunks WHERE entity = ?", (entity,)
            ).fetchall()
        return sorted(row[0] for row in rows)

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
        with self._lock:
            rows = self._con.execute(
                "SELECT canonical, aliases FROM nodes ORDER BY canonical"
            ).fetchall()
        for canonical, aliases in rows:
            out.setdefault(canonical, canonical)
            for alias in (aliases or "").split(ALIAS_SEP):
                alias = alias.strip()
                if alias:
                    out.setdefault(alias, canonical)
        return out
