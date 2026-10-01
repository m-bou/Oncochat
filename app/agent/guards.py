"""Deterministic answer guards: the "double check" step, without an extra LLM call.

A failed guard on the fast tier triggers ONE retry on the smart tier; a failure on the smart tier yields a
safe "I don't know" fallback. Reasons are stored in the trace for later inspection.
"""

import json
import re
from dataclasses import dataclass, field

from app.agent.resolver import Mentions, Resolver
from app.agent.tools import MISUSE_ERRORS

_DECIMAL = re.compile(r"(?<![\w.])(\d+[.,]\d+)(\s*%)?")


@dataclass
class GuardResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def _numbers_in(obj) -> list[float]:
    found: list[float] = []
    if isinstance(obj, bool):
        return found
    if isinstance(obj, int | float):
        found.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_numbers_in(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_numbers_in(v))
    return found


def ungrounded_numbers(answer: str, tool_results: list[dict]) -> list[str]:
    """Decimal numbers quoted in the answer that do not appear (up to rounding) in any tool output."""
    source = [n for r in tool_results for n in _numbers_in(r["result"])]
    bad = []
    for raw, pct in _DECIMAL.findall(answer):
        value = float(raw.replace(",", "."))
        if pct:
            value /= 100
        decimals = len(raw.replace(",", ".").split(".")[1]) + (2 if pct else 0)
        tolerance = 0.5 * 10**-decimals + 1e-9  # any rounding of the source value (half-up or half-even)
        if not any(abs(s - value) <= tolerance for s in source):
            bad.append(raw + (pct or ""))
    return bad


def _arg_cancers(tool_results: list[dict], resolver: Resolver) -> set[str]:
    out = set()
    for r in tool_results:
        c = (r["args"] or {}).get("cancer")
        for name in c if isinstance(c, list) else [c] if c else []:
            resolved = resolver.resolve_cancer(str(name))
            out.add(resolved or f"?{name}")
    return out


def check(*, answer: str, intent: str, mentions: Mentions, tool_results: list[dict], resolver: Resolver,
          hit_max_iterations: bool, llm_error: str | None = None) -> GuardResult:
    reasons: list[str] = []
    if llm_error:
        reasons.append(f"llm_error: {llm_error}")
        return GuardResult(False, reasons)
    if hit_max_iterations:
        reasons.append("max_tool_iterations_reached")
    if not answer.strip():
        reasons.append("empty_answer")
    # Only cancers absent from the dataset (e.g. "esophageal"): the catalog is authoritative, so an honest
    # "not available" answer needs no tool call, but presenting data does not make sense.
    unknown_only = bool(mentions.unknown_cancers) and not mentions.cancers
    if intent == "data" and not tool_results and not unknown_only:
        reasons.append("no_tool_called_for_data_question")
    if tool_results and all(r["result"].get("error") in MISUSE_ERRORS for r in tool_results):
        reasons.append("tool_misuse: " + json.dumps([r["result"].get("error") for r in tool_results]))

    # Substitution: the model queried a cancer the user never mentioned (e.g. esophageal -> gastric).
    if mentions.cancers or mentions.unknown_cancers:
        queried = {c for c in _arg_cancers(tool_results, resolver) if not c.startswith("?")}
        unexpected = queried - set(mentions.cancers)
        if unexpected:
            reasons.append(f"cancer_substitution: asked {mentions.cancers + mentions.unknown_cancers}, "
                           f"queried {sorted(unexpected)}")

    # Values or genes presented for such a question can only come from another indication or an earlier turn.
    if unknown_only:
        checked = any(r["name"] == "list_cancer_types" or r["result"].get("error") == "unknown_cancer"
                      for r in tool_results)
        if tool_results and not checked:  # queried other things, never the cancer the user asked about
            reasons.append(f"unknown_cancer_not_checked: {mentions.unknown_cancers}")
        foreign_genes = set(resolver.extract(answer).genes) - set(mentions.genes)
        if (not mentions.genes and _DECIMAL.search(answer)) or foreign_genes:
            reasons.append(f"data_for_unknown_cancer: {mentions.unknown_cancers}")

    bad = ungrounded_numbers(answer, tool_results)
    if bad:
        reasons.append(f"ungrounded_numbers: {bad[:5]}")
    return GuardResult(not reasons, reasons)
