"""Smoke test of the four brief questions against a real Ollama. Skipped by default; run with:
    uv run pytest -m live
"""

import time
import uuid

import pytest

from app.agent.graph import build_graph
from app.config import Settings
from app.main import build_deps
from app.store.db import Store

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def graph_and_deps(tmp_path_factory):
    settings = Settings(cache_enabled=False)
    deps = build_deps(settings, store=Store(tmp_path_factory.mktemp("live") / "live.db"))
    return build_graph(deps), deps


async def ask(graph_and_deps, question):
    graph, deps = graph_and_deps
    conv = uuid.uuid4().hex
    deps.store.ensure_conversation(conv, question)
    return await graph.ainvoke({"request_id": uuid.uuid4().hex, "conversation_id": conv, "question": question,
                                "history": [], "started_at": time.perf_counter(), "steps": []})


async def test_brief_questions(graph_and_deps):
    out = await ask(graph_and_deps, "How can you help me?")
    assert out["status"] == "help"

    out = await ask(graph_and_deps, "What are the main genes involved in lung cancer?")
    cells = {c for t in out["data"] for row in t["rows"] for c in row}
    assert out["status"] == "ok" and {"ALK", "RET", "ROS1", "STK11", "KRAS"} <= cells

    out = await ask(graph_and_deps, "What is the median value expression of genes involved in breast cancer?")
    assert out["status"] == "ok" and any(["TP53", 0.233] in t["rows"] for t in out["data"])

    out = await ask(graph_and_deps, "What is the median value expression of genes involved in esophageal cancer?")
    assert out["status"] in ("ok", "fallback") and out["data"] == []
