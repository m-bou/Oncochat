from app.store.db import Store


def test_cache_trigger_purges_expired_entries(tmp_path):
    store = Store(tmp_path / "t.db")
    store.cache_set("old", {"answer": "stale"}, ttl_s=-1)  # already expired
    store.cache_set("live", {"answer": "fresh"}, ttl_s=60)  # this insert fires the purge trigger
    keys = [r[0] for r in store._conn.execute("SELECT key FROM cache")]
    assert keys == ["live"]
    assert store.cache_get("live") == {"answer": "fresh"}


def test_schema_is_idempotent(tmp_path):
    path = tmp_path / "t.db"
    Store(path).ensure_conversation("c1", "hello")
    reopened = Store(path)  # schema.sql applied a second time: no error, data kept
    assert reopened.list_conversations()[0]["id"] == "c1"
