"""Shared fixtures. The LLM is replaced by a scripted fake so the graph is tested deterministically and
offline; live Ollama behaviour is covered by eval/run_eval.py (and tests marked `live`)."""

import itertools
from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage

from app.agent.resolver import Resolver
from app.agent.tools import Toolbox
from app.config import Settings
from app.data.repository import Repository
from app.main import build_deps
from app.store.db import Store

_ids = itertools.count()


def call(name: str, **args) -> AIMessage:
    """An AI message requesting one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}"}])


def say(text: str) -> AIMessage:
    return AIMessage(content=text, usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})


class ScriptedLLM:
    """Returns the scripted messages in order; a step may be a callable(messages) -> AIMessage or an Exception."""

    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self.script:
            raise AssertionError("LLM called more times than scripted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(messages) if isinstance(step, Callable) else step


class FakeFactory:
    def __init__(self, fast: list | None = None, smart: list | None = None):
        self.llms = {"fast": ScriptedLLM(fast or []), "smart": ScriptedLLM(smart or [])}

    def get(self, tier: str) -> ScriptedLLM:
        return self.llms[tier]

    def model_name(self, tier: str) -> str:
        return f"fake-{tier}"

    def models(self) -> tuple[str, ...]:
        return ("fake-fast", "fake-smart")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(db_path=tmp_path / "test.db", ollama_host="http://127.0.0.1:9")


@pytest.fixture
def repo(settings) -> Repository:
    return Repository(settings.data_path)


@pytest.fixture
def resolver(repo) -> Resolver:
    return Resolver(repo.catalog)


@pytest.fixture
def toolbox(repo, resolver) -> Toolbox:
    return Toolbox(repo, resolver)


@pytest.fixture
def make_deps(settings):
    def _make(fast: list | None = None, smart: list | None = None):
        return build_deps(settings, llm_factory=FakeFactory(fast, smart), store=Store(settings.db_path))

    return _make
