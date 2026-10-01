"""FastAPI entry point: chat (JSON + SSE streaming), history/traces, health, static UI."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent import prompts
from app.agent.graph import Deps, build_graph, final_payload
from app.agent.llm import OllamaFactory
from app.agent.model_selector import RuleBasedSelector
from app.agent.resolver import Resolver
from app.agent.tools import Toolbox
from app.config import Settings, get_settings
from app.data.repository import Repository
from app.store.db import Store

STATIC = Path(__file__).parent / "static"
log = logging.getLogger("oncochat")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


def build_deps(settings: Settings, llm_factory=None, store: Store | None = None) -> Deps:
    repo = Repository(settings.data_path)
    resolver = Resolver(repo.catalog)
    toolbox = Toolbox(repo, resolver)
    return Deps(settings=settings, repo=repo, resolver=resolver, toolbox=toolbox,
                selector=RuleBasedSelector(settings.complexity_threshold),
                llm_factory=llm_factory or OllamaFactory(settings, toolbox.schemas()),
                store=store or Store(settings.db_path))


async def warmup(d: Deps, status: dict) -> None:
    """Load both models into RAM and prefill Ollama's KV cache with the (request-independent) system prompt and
    tool schemas, so the first real question only pays for its own tokens (docs/ARCHITECTURE.md,
    "Prompt layout")."""
    msgs = [SystemMessage(prompts.system_prompt(d.repo.catalog)), HumanMessage("Hello")]
    for tier in ("fast", "smart"):
        status[tier] = "loading"
        t0 = time.perf_counter()
        try:
            await d.llm_factory.get(tier).ainvoke(msgs)
            status[tier] = f"ready ({time.perf_counter() - t0:.0f}s)"
        except Exception as e:  # noqa: BLE001 - the app still works, the first question is just slower
            status[tier] = f"failed: {type(e).__name__}"
        log.warning("warm-up %s: %s", tier, status[tier])


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.deps = deps or build_deps(get_settings())
        app.state.graph = build_graph(app.state.deps)
        app.state.warmup = {}
        task = None
        if app.state.deps.settings.llm_warmup and isinstance(app.state.deps.llm_factory, OllamaFactory):
            task = asyncio.create_task(warmup(app.state.deps, app.state.warmup))
        yield
        if task:
            task.cancel()

    app = FastAPI(title="OncoChat", version="0.1.0", lifespan=lifespan,
                  description="Natural-language agent over cancer gene-expression primitives.")

    def _input(req: ChatRequest) -> dict:
        d: Deps = app.state.deps
        conv_id = req.conversation_id or uuid.uuid4().hex
        d.store.ensure_conversation(conv_id, req.question.strip())
        history = [{"question": t["question"], "answer": t["answer"]}
                   for t in d.store.conversation_turns(conv_id) if t.get("status") != "error"]
        return {"request_id": uuid.uuid4().hex, "conversation_id": conv_id, "question": req.question.strip(),
                "history": history, "started_at": time.perf_counter(), "steps": []}

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> dict:
        state = await app.state.graph.ainvoke(_input(req))
        return final_payload(state)

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        inp = _input(req)

        async def events() -> AsyncIterator[str]:
            def sse(event: str, data: dict) -> str:
                return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

            yield sse("start", {"request_id": inp["request_id"], "conversation_id": inp["conversation_id"]})
            try:
                async for mode, chunk in app.state.graph.astream(inp, stream_mode=["custom", "messages"]):
                    if mode == "custom":
                        yield sse(chunk.get("type", "step"), chunk)
                    elif mode == "messages":
                        msg, meta = chunk
                        if (meta.get("langgraph_node") == "agent" and isinstance(msg, AIMessageChunk)
                                and isinstance(msg.content, str) and msg.content and not msg.tool_call_chunks):
                            yield sse("token", {"text": msg.content})
            except Exception as e:  # noqa: BLE001
                yield sse("error", {"message": f"{type(e).__name__}: {e}"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/conversations")
    async def conversations(limit: int = 50) -> list[dict]:
        return app.state.deps.store.list_conversations(limit)

    @app.get("/api/conversations/{conversation_id}")
    async def conversation(conversation_id: str) -> list[dict]:
        return app.state.deps.store.conversation_turns(conversation_id)

    @app.get("/api/history")
    async def history(limit: int = 100, status: str | None = None) -> dict:
        store: Store = app.state.deps.store
        return {"stats": store.stats(), "requests": store.history(limit, status)}

    @app.get("/api/traces/{request_id}")
    async def trace(request_id: str) -> dict:
        t = app.state.deps.store.trace(request_id)
        if t is None:
            raise HTTPException(404, "unknown request_id")
        return t

    @app.delete("/api/cache")
    async def clear_cache() -> dict:
        return {"deleted": app.state.deps.store.cache_clear()}

    @app.get("/api/health")
    async def health() -> dict:
        d: Deps = app.state.deps
        s = d.settings
        out = {"status": "ok", "ollama_host": s.ollama_host, "models": {"fast": s.fast_model, "smart": s.smart_model},
               "data_version": d.repo.catalog.data_version, "prompt_version": prompts.PROMPT_VERSION,
               "warmup": app.state.warmup,
               "cancers": list(d.repo.catalog.cancers)}
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                tags = (await client.get(f"{s.ollama_host}/api/tags")).json()
            available = {m["name"] for m in tags.get("models", [])}
            missing = [m for m in (s.fast_model, s.smart_model)
                       if m not in available and f"{m}:latest" not in available]
            out["ollama"] = "ok" if not missing else f"missing models: {missing}"
        except Exception as e:  # noqa: BLE001
            out["ollama"] = f"unreachable: {type(e).__name__}"
        if out["ollama"] != "ok":
            out["status"] = "degraded"
        return out

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
