"""Live evaluation against Ollama: behaviour pass-rate, tier usage, escalation rate, latency percentiles.

    uv run python -m eval.run_eval                 # adaptive selection (production behaviour)
    uv run python -m eval.run_eval --tier fast     # force a tier, to calibrate the complexity threshold
    uv run python -m eval.run_eval --only brief    # filter case ids by substring
    uv run python -m eval.run_eval --repeat 3      # repeat to measure non-determinism

Results are written to eval/results/<timestamp>.json (the cache is disabled during evaluation).
"""

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import yaml

from app.agent.graph import build_graph
from app.agent.model_selector import Decision, SelectionInput
from app.config import get_settings
from app.main import build_deps, warmup
from app.store.db import Store

HERE = Path(__file__).parent


class FixedSelector:
    def __init__(self, tier: str):
        self.tier = tier

    def select(self, inp: SelectionInput) -> Decision:
        return Decision(tier=self.tier, score=0.0, reason=f"forced {self.tier}")


def evaluate(case: dict, out: dict) -> list[str]:
    exp = case["expect"]
    failures: list[str] = []
    answer = out.get("answer") or ""
    tables = out.get("data") or []
    table_text = json.dumps(tables)
    cells = [str(c) for t in tables for row in t["rows"] for c in row]
    called = {t["name"] for t in out.get("tool_results") or []}

    if out.get("status") not in exp.get("status", ["ok"]):
        failures.append(f"status={out.get('status')} guard={out.get('guard', {}).get('reasons')}")
    if (tools := exp.get("tools_any")) and not called & set(tools):
        failures.append(f"tools {sorted(called)} ∩ {tools} = ∅")
    for item in exp.get("table_contains", []):
        if item not in cells:
            failures.append(f"table missing {item}")
    for gene, value in (exp.get("table_values") or {}).items():
        if not any(t["rows"] and [gene, value] in t["rows"] for t in tables):
            failures.append(f"table missing {gene}={value}")
    if (any_of := exp.get("answer_contains_any")) and not any(s.lower() in answer.lower() for s in any_of):
        failures.append(f"answer lacks any of {any_of}")
    for s in exp.get("answer_contains_all", []):
        if s.lower() not in answer.lower():
            failures.append(f"answer lacks '{s}'")
    for s in exp.get("must_not_contain", []):
        if s.lower() in (answer + table_text).lower():
            failures.append(f"forbidden '{s}' present")
    if exp.get("no_tables") and tables:
        failures.append("tables shown but none expected")
    return failures


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(round(q * (len(values) - 1))))], 1)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["fast", "smart"], help="force a model tier (disables adaptive selection)")
    ap.add_argument("--only", help="run only cases whose id contains this substring")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--no-escalation", action="store_true")
    args = ap.parse_args()

    settings = get_settings().model_copy(update={"cache_enabled": False,
                                                 "escalation_enabled": not args.no_escalation})
    deps = build_deps(settings, store=Store(HERE / "results" / "eval.db"))
    if args.tier:
        deps.selector = FixedSelector(args.tier)
    graph = build_graph(deps)
    status: dict = {}
    await warmup(deps, status)  # same as the app at startup: latencies below are warm-model latencies
    print("warm-up:", status, flush=True)
    cases = [c for c in yaml.safe_load((HERE / "questions.yml").read_text(encoding="utf-8"))
             if not args.only or args.only in c["id"]]

    rows = []
    for rep in range(args.repeat):
        for case in cases:
            conv = uuid.uuid4().hex
            deps.store.ensure_conversation(conv, case["question"])
            t0 = time.perf_counter()
            out = await graph.ainvoke({"request_id": uuid.uuid4().hex, "conversation_id": conv,
                                       "question": case["question"], "history": case.get("history", []),
                                       "started_at": t0, "steps": []})
            ms = round((time.perf_counter() - t0) * 1000, 1)
            failures = evaluate(case, out)
            rows.append({"id": case["id"], "rep": rep, "pass": not failures, "failures": failures, "latency_ms": ms,
                         "status": out.get("status"), "tier": out.get("tier"), "escalated": bool(out.get("escalated")),
                         "tools": [(t["name"], t["args"]) for t in out.get("tool_results") or []],
                         "answer": out.get("answer"), "request_id": out["request_id"]})
            mark = "PASS" if not failures else "FAIL"
            print(f"[{mark}] {case['id']:<22} {ms:>8.0f} ms  tier={out.get('tier') or '-':<5} "
                  f"esc={int(bool(out.get('escalated')))}  {'; '.join(failures)}", flush=True)

    llm_rows = [r for r in rows if r["tier"]]
    by_tier = {t: [r["latency_ms"] for r in llm_rows if r["tier"] == t] for t in ("fast", "smart")}
    summary = {
        "n": len(rows), "pass_rate": round(sum(r["pass"] for r in rows) / len(rows), 3),
        "escalation_rate": round(sum(r["escalated"] for r in llm_rows) / max(1, len(llm_rows)), 3),
        "tier_counts": {t: len(v) for t, v in by_tier.items()},
        "latency_ms": {t: {"p50": pct(v, 0.5), "p95": pct(v, 0.95),
                           "mean": round(statistics.fmean(v), 1) if v else None} for t, v in by_tier.items()},
        "models": {"fast": settings.fast_model, "smart": settings.smart_model},
        "forced_tier": args.tier, "escalation": not args.no_escalation,
        "data_version": deps.repo.catalog.data_version,
    }
    print(json.dumps(summary, indent=2))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{stamp}{'-' + args.tier if args.tier else ''}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
