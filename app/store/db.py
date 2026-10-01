"""SQLite persistence: conversations, request traces, per-node steps and the answer cache.

History is queried by id, time and status, so a single SQLite file is all it needs.
The schema lives in schema.sql (one comment per column) and is applied idempotently at startup; a single
connection guarded by a lock is plenty for a POC.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")  # documented schema, also the data-model reference

_JSON_COLS = ("features_json", "guard_json", "tool_calls_json", "data_json")


def _dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ conversations
    def ensure_conversation(self, conversation_id: str, title: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO conversations(id, title, created_at) VALUES (?,?,?)",
                               (conversation_id, title[:80], time.time()))

    def list_conversations(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT c.id, c.title, c.created_at, MAX(r.created_at) AS updated_at, COUNT(r.id) AS n_messages "
            "FROM conversations c LEFT JOIN requests r ON r.conversation_id = c.id "
            "GROUP BY c.id ORDER BY COALESCE(MAX(r.created_at), c.created_at) DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def conversation_turns(self, conversation_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM requests WHERE conversation_id = ? ORDER BY created_at", (conversation_id,))
        return [self._decode(r) for r in rows]

    # ------------------------------------------------------------------ traces
    def save_request(self, record: dict, steps: list[dict]) -> None:
        rec = {k: (_dumps(v) if k in _JSON_COLS else v) for k, v in record.items()}
        rec.setdefault("created_at", time.time())
        cols = ", ".join(rec)
        marks = ", ".join("?" for _ in rec)
        with self._lock, self._conn:
            self._conn.execute(f"INSERT OR REPLACE INTO requests({cols}) VALUES ({marks})", tuple(rec.values()))
            self._conn.executemany(
                "INSERT INTO steps(request_id, seq, node, latency_ms, detail_json) VALUES (?,?,?,?,?)",
                [(record["id"], i, s["node"], s.get("latency_ms"), _dumps(s.get("detail")))
                 for i, s in enumerate(steps)])

    def history(self, limit: int = 100, status: str | None = None) -> list[dict]:
        sql = ("SELECT id, conversation_id, question, answer, status, intent, tier, model, escalated, cache_hit, "
               "complexity_score, latency_ms, prompt_tokens, completion_tokens, created_at FROM requests")
        params: tuple = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        rows = self._conn.execute(sql + " ORDER BY created_at DESC LIMIT ?", (*params, limit))
        return [dict(r) for r in rows]

    def trace(self, request_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        if row is None:
            return None
        out = self._decode(row)
        out["steps"] = [
            {"seq": s["seq"], "node": s["node"], "latency_ms": s["latency_ms"],
             "detail": json.loads(s["detail_json"]) if s["detail_json"] else None}
            for s in self._conn.execute("SELECT * FROM steps WHERE request_id = ? ORDER BY seq", (request_id,))
        ]
        return out

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*) n, AVG(latency_ms) avg_ms, SUM(cache_hit) cache_hits, SUM(escalated) escalations, "
            "SUM(status='fallback') fallbacks, SUM(status='error') errors, SUM(tier='smart') smart, "
            "SUM(tier='fast') fast FROM requests").fetchone()
        return dict(row)

    # ------------------------------------------------------------------ cache
    def cache_get(self, key: str) -> dict | None:
        row = self._conn.execute("SELECT payload_json, expires_at FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None or row["expires_at"] < time.time():
            return None
        with self._lock, self._conn:
            self._conn.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
        return json.loads(row["payload_json"])

    def cache_set(self, key: str, payload: dict, ttl_s: int) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO cache(key, payload_json, created_at, expires_at) "
                               "VALUES (?,?,?,?)", (key, _dumps(payload), now, now + ttl_s))

    def cache_clear(self) -> int:
        with self._lock, self._conn:
            return self._conn.execute("DELETE FROM cache").rowcount

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict:
        d = dict(row)
        for k in _JSON_COLS:
            if d.get(k):
                d[k.removesuffix("_json")] = json.loads(d.pop(k))
            else:
                d.pop(k, None)
        return d
