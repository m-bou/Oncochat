"""LangGraph orchestration (see docs/ARCHITECTURE.md for the diagram).

START → cache_lookup ─hit→ persist
          └→ router ─help/greeting→ capabilities → persist
               └→ select_model → agent ⇄ tools → guard ─ok→ answer → cache_store → persist
                                   ↑                  ├─fail & fast→ escalate ─┘(agent, smart tier)
                                   └──────────────────┘└─fail & smart→ fallback → persist
"""

import json
import operator
import time
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agent import guards, prompts
from app.agent.model_selector import ModelSelector, SelectionInput
from app.agent.resolver import Mentions, Resolver
from app.agent.router import route
from app.agent.tools import Toolbox, to_table
from app.config import Settings
from app.data.repository import Repository
from app.store.cache import cache_key
from app.store.db import Store
from app.store.tracing import emit, traced


class AgentState(TypedDict, total=False):
    # inputs
    request_id: str
    conversation_id: str
    question: str
    history: list[dict]  # previous turns: [{"question": ..., "answer": ...}]
    started_at: float
    # routing / selection
    mentions: Mentions
    intent: str
    lang: str
    cache_key: str | None
    cache_hit: bool
    decision: dict
    tier: str
    model: str
    escalated: bool
    # agent loop (messages are overwritten, not appended: escalation restarts from scratch)
    messages: list[BaseMessage]
    tool_results: list[dict]
    iterations: int
    llm_error: str | None
    usage: dict
    # outputs
    guard: dict
    answer: str
    data: list[dict]
    status: str
    steps: Annotated[list[dict], operator.add]


@dataclass
class Deps:
    settings: Settings
    repo: Repository
    resolver: Resolver
    toolbox: Toolbox
    selector: ModelSelector
    llm_factory: Any  # OllamaFactory-like: .get(tier) -> Runnable, .model_name(tier), .models()
    store: Store


def _last_ai(state: AgentState) -> AIMessage | None:
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage):
            return m
    return None


def _text(msg: AIMessage | None) -> str:
    if msg is None:
        return ""
    c = msg.content
    if isinstance(c, list):  # some providers return content blocks
        c = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c).strip()


def build_graph(deps: Deps):
    s, toolbox, resolver = deps.settings, deps.toolbox, deps.resolver
    catalog = deps.repo.catalog

    # ------------------------------------------------------------------ nodes
    @traced("cache_lookup")
    async def cache_lookup(state: AgentState) -> dict:
        if not s.cache_enabled or state.get("history"):
            return {"cache_key": None, "cache_hit": False, "_detail": {"cacheable": False}}
        key = cache_key(state["question"], data_version=catalog.data_version,
                        prompt_version=prompts.PROMPT_VERSION, models=deps.llm_factory.models())
        hit = deps.store.cache_get(key)
        if hit is None:
            return {"cache_key": key, "cache_hit": False, "_detail": {"cacheable": True, "hit": False}}
        return {"cache_key": key, "cache_hit": True, "status": "cached", "answer": hit["answer"],
                "data": hit.get("data", []), "tier": hit.get("tier"), "model": hit.get("model"),
                "intent": hit.get("intent"), "lang": hit.get("lang"), "_detail": {"cacheable": True, "hit": True}}

    @traced("router")
    async def router(state: AgentState) -> dict:
        mentions = resolver.extract(state["question"])
        r = route(state["question"], mentions)
        return {"mentions": mentions, "intent": r.intent, "lang": r.lang,
                "_detail": {"intent": r.intent, "lang": r.lang, "mentions": asdict(mentions)}}

    @traced("capabilities")
    async def capabilities(state: AgentState) -> dict:
        text = prompts.help_answer(state["lang"], catalog, list(toolbox.specs.values()),
                                   greeting=state["intent"] == "greeting")
        return {"answer": text, "status": "help", "data": [], "_detail": {"llm": False}}

    @traced("select_model")
    async def select_model(state: AgentState) -> dict:
        d = deps.selector.select(SelectionInput(state["question"], state["mentions"], bool(state.get("history"))))
        decision = asdict(d)
        return {"decision": decision, "tier": d.tier, "escalated": False, "messages": [], "tool_results": [],
                "iterations": 0, "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "_detail": {"tier": d.tier, "model": deps.llm_factory.model_name(d.tier), "reason": d.reason}}

    def _initial_messages(state: AgentState) -> list[BaseMessage]:
        msgs: list[BaseMessage] = [SystemMessage(prompts.system_prompt(catalog))]  # stable, KV-cached prefix
        for turn in (state.get("history") or [])[-s.history_turns:]:
            msgs += [HumanMessage(turn["question"]), AIMessage(turn["answer"] or "")]
        msgs.append(HumanMessage(state["question"] + prompts.hint(state["mentions"], state["lang"], state["intent"])))
        return msgs

    @traced("agent")
    async def agent(state: AgentState) -> dict:
        tier = state["tier"]
        msgs = state.get("messages") or _initial_messages(state)
        model = deps.llm_factory.model_name(tier)
        try:
            ai = await deps.llm_factory.get(tier).ainvoke(msgs)
        except Exception as e:  # noqa: BLE001 - surfaced to guard -> escalate/fallback, recorded in trace
            err = f"{type(e).__name__}: {e}"[:300]
            return {"llm_error": err, "messages": msgs, "model": model, "_detail": {"model": model, "error": err}}
        usage = dict(state.get("usage") or {})
        meta = getattr(ai, "usage_metadata", None) or {}
        usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + meta.get("input_tokens", 0)
        usage["completion_tokens"] = usage.get("completion_tokens", 0) + meta.get("output_tokens", 0)
        detail = {"model": model, "tier": tier, "tool_calls": [{"name": c["name"], "args": c["args"]}
                                                               for c in ai.tool_calls]}
        if not ai.tool_calls:
            detail["answer_preview"] = _text(ai)[:200]
        return {"messages": [*msgs, ai], "iterations": state.get("iterations", 0) + 1, "model": model,
                "llm_error": None, "usage": usage, "_detail": detail}

    @traced("tools")
    async def tools(state: AgentState) -> dict:
        ai = _last_ai(state)
        results = list(state.get("tool_results") or [])
        new_msgs: list[BaseMessage] = []
        detail = []
        for call in ai.tool_calls:
            result = toolbox.execute(call["name"], call["args"])
            results.append({"name": call["name"], "args": call["args"], "result": result})
            new_msgs.append(ToolMessage(json.dumps(result, ensure_ascii=False), tool_call_id=call["id"] or "",
                                        name=call["name"]))
            detail.append({"name": call["name"], "args": call["args"], "error": result.get("error")})
        return {"messages": [*state["messages"], *new_msgs], "tool_results": results, "_detail": detail}

    @traced("guard")
    async def guard(state: AgentState) -> dict:
        ai = _last_ai(state)
        hit_max = bool(ai and ai.tool_calls)  # still wants tools but iteration budget exhausted
        answer = "" if hit_max else _text(ai)
        g = guards.check(answer=answer, intent=state["intent"], mentions=state["mentions"],
                         tool_results=state.get("tool_results") or [], resolver=resolver,
                         hit_max_iterations=hit_max, llm_error=state.get("llm_error"))
        gd = {"ok": g.ok, "reasons": g.reasons, "tier": state["tier"]}
        return {"guard": gd, "answer": answer, "_detail": gd}

    @traced("escalate")
    async def escalate(state: AgentState) -> dict:
        """Retry on the smart tier. The rejected answer and a plain-language critique built from the guard
        reasons are shown to the model (deterministic critic, LLM reviser), unless the failure was an LLM
        error, in which case the retry starts clean."""
        emit({"type": "reset"})  # UI drops the streamed draft of the failed attempt
        reasons = state["guard"]["reasons"]
        msgs: list[BaseMessage] = []
        feedback = None
        if not state.get("llm_error"):
            feedback = prompts.critique(reasons)
            msgs = [*_initial_messages(state), AIMessage(state.get("answer") or "(no answer)"),
                    HumanMessage(feedback)]
        return {"tier": "smart", "escalated": True, "messages": msgs, "tool_results": [], "iterations": 0,
                "llm_error": None,
                "_detail": {"from": state["tier"], "to": "smart", "reasons": reasons, "feedback": feedback}}

    @traced("answer")
    async def answer(state: AgentState) -> dict:
        tables, seen = [], set()
        for r in state.get("tool_results") or []:
            out = to_table(r["name"], r["args"] or {}, r["result"])
            for t in out if isinstance(out, list) else [out]:
                if t and (sig := json.dumps(t, sort_keys=True)) not in seen:
                    seen.add(sig)
                    tables.append(t)
        return {"status": "ok", "data": tables, "_detail": {"tables": len(tables)}}

    @traced("fallback")
    async def fallback(state: AgentState) -> dict:
        lang = state.get("lang", "en")
        if state.get("llm_error"):
            text = prompts.LLM_DOWN[lang].format(error=state["llm_error"].split(":")[0])
            status = "error"
        else:
            text = prompts.FALLBACK[lang].format(cancers=", ".join(catalog.cancers))
            status = "fallback"
        emit({"type": "reset"})
        return {"answer": text, "status": status, "data": [], "_detail": {"status": status}}

    @traced("cache_store")
    async def cache_store(state: AgentState) -> dict:
        key = state.get("cache_key")
        if not key:
            return {"_detail": {"stored": False}}
        deps.store.cache_set(key, {"answer": state["answer"], "data": state.get("data", []),
                                   "tier": state.get("tier"), "model": state.get("model"),
                                   "intent": state.get("intent"), "lang": state.get("lang")}, s.cache_ttl_s)
        return {"_detail": {"stored": True}}

    async def persist(state: AgentState) -> dict:
        latency_ms = round((time.perf_counter() - state["started_at"]) * 1000, 1)
        decision = state.get("decision") or {}
        usage = state.get("usage") or {}
        record = {
            "id": state["request_id"], "conversation_id": state["conversation_id"],
            "question": state["question"], "answer": state.get("answer", ""), "status": state.get("status"),
            "intent": state.get("intent"), "lang": state.get("lang"), "tier": state.get("tier"),
            "model": state.get("model"), "escalated": int(bool(state.get("escalated"))),
            "cache_hit": int(bool(state.get("cache_hit"))), "complexity_score": decision.get("score"),
            "features_json": decision.get("features"), "guard_json": state.get("guard"),
            "tool_calls_json": [{"name": r["name"], "args": r["args"], "error": r["result"].get("error")}
                                for r in state.get("tool_results") or []],
            "data_json": state.get("data", []), "latency_ms": latency_ms,
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "data_version": catalog.data_version, "prompt_version": prompts.PROMPT_VERSION,
            "error": state.get("llm_error"),
        }
        deps.store.save_request(record, state.get("steps") or [])
        emit({"type": "final", **final_payload(state, latency_ms)})
        return {}

    # ------------------------------------------------------------------ edges
    def after_cache(state: AgentState) -> str:
        return "persist" if state.get("cache_hit") else "router"

    def after_router(state: AgentState) -> str:
        return "capabilities" if state["intent"] in ("help", "greeting") else "select_model"

    def after_agent(state: AgentState) -> str:
        ai = _last_ai(state)
        if state.get("llm_error") or not ai or not ai.tool_calls:
            return "guard"
        return "tools" if state.get("iterations", 0) < s.max_tool_iterations else "guard"

    def after_guard(state: AgentState) -> str:
        if state["guard"]["ok"]:
            return "answer"
        if s.escalation_enabled and state["tier"] == "fast" and not state.get("escalated"):
            return "escalate"
        return "fallback"

    g = StateGraph(AgentState)
    for name, fn in [("cache_lookup", cache_lookup), ("router", router), ("capabilities", capabilities),
                     ("select_model", select_model), ("agent", agent), ("tools", tools), ("guard", guard),
                     ("escalate", escalate), ("answer", answer), ("fallback", fallback),
                     ("cache_store", cache_store), ("persist", persist)]:
        g.add_node(name, fn)
    g.add_edge(START, "cache_lookup")
    g.add_conditional_edges("cache_lookup", after_cache, ["persist", "router"])
    g.add_conditional_edges("router", after_router, ["capabilities", "select_model"])
    g.add_edge("capabilities", "persist")
    g.add_edge("select_model", "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "guard"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("guard", after_guard, ["answer", "escalate", "fallback"])
    g.add_edge("escalate", "agent")
    g.add_edge("answer", "cache_store")
    g.add_edge("cache_store", "persist")
    g.add_edge("fallback", "persist")
    g.add_edge("persist", END)
    return g.compile()


def final_payload(state: dict, latency_ms: float | None = None) -> dict:
    if latency_ms is None:
        latency_ms = round((time.perf_counter() - state["started_at"]) * 1000, 1)
    return {
        "request_id": state.get("request_id"), "conversation_id": state.get("conversation_id"),
        "answer": state.get("answer", ""), "data": state.get("data", []), "status": state.get("status"),
        "intent": state.get("intent"), "lang": state.get("lang"), "tier": state.get("tier"),
        "model": state.get("model"), "escalated": bool(state.get("escalated")),
        "cache_hit": bool(state.get("cache_hit")), "latency_ms": latency_ms,
        "guard": state.get("guard"), "decision": state.get("decision"),
    }
