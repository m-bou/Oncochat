import json
import time
import uuid

import httpx
from fastapi.testclient import TestClient

from app.agent.graph import build_graph
from app.main import create_app
from tests.conftest import call, say


def _input(question, history=None):
    return {"request_id": uuid.uuid4().hex, "conversation_id": "c1", "question": question,
            "history": history or [], "started_at": time.perf_counter(), "steps": []}


async def run(deps, question, history=None):
    deps.store.ensure_conversation("c1", question)
    return await build_graph(deps).ainvoke(_input(question, history))


async def test_help_needs_no_llm(make_deps):
    deps = make_deps()  # empty scripts: any LLM call would fail the test
    out = await run(deps, "How can you help me?")
    assert out["status"] == "help" and "lung" in out["answer"]
    assert [s["node"] for s in out["steps"]] == ["cache_lookup", "router", "capabilities"]


async def test_lung_targets(make_deps):
    deps = make_deps(fast=[call("get_targets", cancer="lung"), say("The genes are ALK, RET, ROS1, STK11, KRAS.")])
    out = await run(deps, "What are the main genes involved in lung cancer?")
    assert out["status"] == "ok" and out["tier"] == "fast" and not out["escalated"]
    assert out["data"][0]["rows"] == [["ALK"], ["RET"], ["ROS1"], ["STK11"], ["KRAS"]]
    trace = deps.store.trace(out["request_id"])
    assert trace["tool_calls"] == [{"name": "get_targets", "args": {"cancer": "lung"}, "error": None}]
    assert trace["prompt_tokens"] == 100


async def test_breast_values_come_from_fixed_primitive(make_deps):
    deps = make_deps(fast=[call("summarize_expressions", cancer="breast"),
                           say("Median of medians is 0.4345; TP53 is 0.233.")])
    out = await run(deps, "What is the median value expression of genes involved in breast cancer?")
    assert out["status"] == "ok", out["guard"]
    rows = dict(out["data"][0]["rows"])
    assert rows["TP53"] == 0.233 and rows["PIK3CA"] == 0.449


async def test_esophageal_unknown(make_deps):
    deps = make_deps(fast=[call("get_targets", cancer="esophageal"),
                           say("Esophageal cancer is not available in the dataset.")])
    out = await run(deps, "What is the median value expression of genes involved in esophageal cancer?")
    assert out["status"] == "ok" and out["data"] == []
    # the deterministic pre-analysis hint is given to the model
    first_call = deps.llm_factory.get("fast").calls[0]
    assert "NOT in the dataset: esophageal" in first_call[-1].content
    assert "Parser notes" not in first_call[0].content  # system prompt stays request-independent (KV cache)


async def test_substitution_escalates_then_smart_answers(make_deps):
    deps = make_deps(
        fast=[call("summarize_expressions", cancer="gastric"), say("Values are 0.453 and 0.834.")],
        smart=[call("get_targets", cancer="esophageal"), say("Esophageal cancer is not in the dataset.")])
    out = await run(deps, "What is the median value expression of genes involved in esophageal cancer?")
    assert out["escalated"] and out["tier"] == "smart" and out["status"] == "ok"
    nodes = [s["node"] for s in out["steps"]]
    assert nodes.count("guard") == 2 and "escalate" in nodes
    # the smart tier sees the rejected answer and a plain-language critique
    retry_prompt = deps.llm_factory.get("smart").calls[0][-1].content
    assert "rejected by an automatic check" in retry_prompt and "did not ask about" in retry_prompt


async def test_hallucinated_numbers_fall_back(make_deps):
    deps = make_deps(fast=[call("get_targets", cancer="lung"), say("ALK is 0.99.")],
                     smart=[call("get_targets", cancer="lung"), say("ALK is 0.98.")])
    out = await run(deps, "What are the main genes involved in lung cancer?")
    assert out["status"] == "fallback" and out["answer"].startswith("I don't know")


async def test_llm_down(make_deps):
    err = httpx.ConnectError("connection refused")
    deps = make_deps(fast=[err], smart=[err])
    out = await run(deps, "What are the main genes involved in lung cancer?")
    assert out["status"] == "error" and "unavailable" in out["answer"]


async def test_cache_hit_on_second_ask(make_deps):
    q = "What are the main genes involved in lung cancer?"
    deps = make_deps(fast=[call("get_targets", cancer="lung"), say("ALK, RET, ROS1, STK11 and KRAS.")])
    first = await run(deps, q)
    second = await run(deps, "  what are the MAIN genes involved in lung cancer ")  # normalised key
    assert not first["cache_hit"] and second["cache_hit"] and second["status"] == "cached"
    assert second["answer"] == first["answer"]


async def test_data_question_reminds_tool_use(make_deps):
    deps = make_deps(fast=[call("get_targets", cancer="lung"), say("ALK, RET, ROS1, STK11 and KRAS.")])
    await run(deps, "Quels sont les gènes du cancer du poumon ?",
              history=[{"question": "breast?", "answer": "ESR1 is 0.716."}])
    last = deps.llm_factory.get("fast").calls[0][-1].content
    assert "call a tool first" in last and "Answer in French" in last


async def test_followup_uses_history_and_skips_cache(make_deps):
    deps = make_deps(fast=[call("get_expressions", genes=["KRAS"], cancer="lung"), say("KRAS is 0.359.")])
    out = await run(deps, "What is the expression of KRAS in it?",
                    history=[{"question": "Genes in lung cancer?", "answer": "ALK, RET, ROS1, STK11, KRAS."}])
    assert out["status"] == "ok" and out["cache_key"] is None
    msgs = deps.llm_factory.get("fast").calls[0]
    assert [m.type for m in msgs] == ["system", "human", "ai", "human"]


# ---------------------------------------------------------------------------------------------- API
def test_api_end_to_end(make_deps):
    deps = make_deps(fast=[call("get_targets", cancer="lung"), say("ALK, RET, ROS1, STK11 and KRAS.")])
    with TestClient(create_app(deps)) as client:
        r = client.post("/api/chat/stream", json={"question": "What are the main genes involved in lung cancer?"})
        assert r.status_code == 200
        events = [(b.split("\n")[0].removeprefix("event: "), json.loads(b.split("\n")[1].removeprefix("data: ")))
                  for b in r.text.strip().split("\n\n")]
        kinds = [e for e, _ in events]
        assert kinds[0] == "start" and kinds[-1] == "final" and "step" in kinds
        final = events[-1][1]
        assert final["status"] == "ok" and final["tier"] == "fast"

        conv = client.get(f"/api/conversations/{final['conversation_id']}").json()
        assert conv[0]["answer"] == final["answer"]
        assert client.get("/api/conversations").json()[0]["n_messages"] == 1
        hist = client.get("/api/history").json()
        assert hist["stats"]["n"] == 1
        trace = client.get(f"/api/traces/{final['request_id']}").json()
        assert [s["node"] for s in trace["steps"]][:3] == ["cache_lookup", "router", "select_model"]

        r = client.post("/api/chat", json={"question": "How can you help me?"})
        assert r.json()["status"] == "help"

        health = client.get("/api/health").json()
        assert health["status"] == "degraded" and health["ollama"].startswith("unreachable")
        assert client.get("/").status_code == 200
