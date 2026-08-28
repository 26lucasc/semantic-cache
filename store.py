"""Vector store: SQLite for durability + an in-memory matrix for search.

WHY BRUTE FORCE: nearest-neighbor search here is one matrix multiply. At 100k
cached entries with 384-dim vectors that's ~40M float ops -- around 5ms in
NumPy. An approximate index (sqlite-vec, pgvector/HNSW, Qdrant) only starts
paying for itself in the millions. Swapping it in later touches ONLY
`search()`; nothing else in the codebase knows how search works.
"""
import sqlite3
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id         INTEGER PRIMARY KEY,
    tenant     TEXT NOT NULL,       -- cache is partitioned per tenant; see cache.py
    prompt     TEXT NOT NULL,
    query_hash TEXT NOT NULL,     -- sha256 of normalized prompt; Layer 1 lookup
    answer     TEXT NOT NULL,
    vector     BLOB NOT NULL,       -- float32, L2-normalized
    created_at REAL NOT NULL,
    last_hit   REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tenant ON entries(tenant);
-- Layer 1 is a real index lookup, not a scan. The reference implementation
-- this was compared against loops over every key doing one round trip each
-- and still calls it O(1); it is not.
CREATE UNIQUE INDEX IF NOT EXISTS idx_hash ON entries(tenant, query_hash);
"""


class Store:
    def __init__(self, path: str = "cache.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._load()

    def _load(self):
        """Pull every vector into one contiguous matrix for fast search."""
        rows = self.db.execute(
            "SELECT id, tenant, vector FROM entries ORDER BY id"
        ).fetchall()
        self.ids = [r[0] for r in rows]
        self.tenants = [r[1] for r in rows]
        self.matrix = (
            np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            if rows else np.zeros((0, 0), dtype=np.float32)
        )

    def add(self, tenant: str, prompt: str, qhash: str, answer: str,
            vector: np.ndarray) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT OR REPLACE INTO entries"
            " (tenant, prompt, query_hash, answer, vector, created_at, last_hit)"
            " VALUES (?,?,?,?,?,?,?)",
            (tenant, prompt, qhash, answer,
             vector.astype(np.float32).tobytes(), now, now),
        )
        self.db.commit()
        self._load()   # TODO: append to the matrix instead of reloading it all
        return cur.lastrowid

    def get_by_hash(self, tenant: str, qhash: str):
        """Layer 1: exact match. One indexed lookup, no embedding call, and
        no false-hit risk -- the normalized strings are identical."""
        row = self.db.execute(
            "SELECT id FROM entries WHERE tenant=? AND query_hash=?", (tenant, qhash)
        ).fetchone()
        return row[0] if row else None

    def search(self, tenant: str, vector: np.ndarray):
        """Return (entry_id, similarity) for the closest entry in this tenant.

        Returns (None, 0.0) on an empty cache -- the cold-start case.
        """
        if not self.ids:
            return None, 0.0

        sims = self.matrix @ vector.astype(np.float32)   # all cosines at once

        # Tenant isolation: an entry from another tenant must never be
        # reachable, so mask it out rather than filtering after the argmax.
        mask = np.array([t == tenant for t in self.tenants])
        if not mask.any():
            return None, 0.0
        sims = np.where(mask, sims, -1.0)

        best = int(np.argmax(sims))
        return self.ids[best], float(sims[best])

    def search_topk(self, tenant: str, vector: np.ndarray, k: int = 5,
                    floor: float = 0.15):
        """Layer 2 retrieval for the two-stage design: return up to k
        candidates, loosely.

        Deliberately permissive. Once a verifier exists, a TIGHT retriever is
        the liability -- anything it drops never reaches the verifier and is
        unrecoverable. `floor` exists only to avoid verifying garbage on a
        cold cache, not to make decisions.
        """
        if not self.ids:
            return []
        sims = self.matrix @ vector.astype(np.float32)
        mask = np.array([t == tenant for t in self.tenants])
        if not mask.any():
            return []
        sims = np.where(mask, sims, -1.0)
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order if sims[i] >= floor]

    def get(self, entry_id: int):
        row = self.db.execute(
            "SELECT prompt, answer, created_at, hits FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        return {"prompt": row[0], "answer": row[1], "created_at": row[2], "hits": row[3]}

    def record_hit(self, entry_id: int):
        self.db.execute(
            "UPDATE entries SET hits = hits + 1, last_hit = ? WHERE id = ?",
            (time.time(), entry_id),
        )
        self.db.commit()

    def evict_expired(self, ttl_seconds: float):
        """TODO(you): staleness. A cached 'refunds take 30 days' outlives the
        policy change that made it 14. TTL is the crude fix; an explicit
        invalidate-by-topic endpoint is the real one."""
        raise NotImplementedError

    def evict_lru(self, max_entries: int):
        """TODO(you): unbounded growth. Drop least-recently-hit entries."""
        raise NotImplementedError

    def stats(self):
        row = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM entries"
        ).fetchone()
        return {"entries": row[0], "total_hits": row[1]}
