"""Adaptive model selection: deterministic complexity scoring + cascade escalation.

Why not an LLM classifier? On CPU every LLM call costs seconds; spending one to decide which model to use
defeats the purpose. Features are cheap, explainable (stored in each trace) and calibrated offline with the
eval harness (`eval/run_eval.py --tier fast|smart` shows where the fast tier breaks).

Extending: add a `Feature` to `DEFAULT_FEATURES` (e.g. `n_tables_referenced` once the data grows), or
replace `RuleBasedSelector` by any object implementing `ModelSelector.select`.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from app.agent.resolver import Mentions, normalize

Tier = Literal["fast", "smart"]


@dataclass(frozen=True)
class SelectionInput:
    question: str
    mentions: Mentions
    has_history: bool


@dataclass(frozen=True)
class Feature:
    name: str
    weight: float
    extract: Callable[[SelectionInput], float]


@dataclass
class Decision:
    tier: Tier
    score: float
    features: dict[str, float] = field(default_factory=dict)
    reason: str = ""


class ModelSelector(Protocol):
    def select(self, inp: SelectionInput) -> Decision: ...


_COMPARISON = re.compile(r"\b(compare|comparison|versus|vs\.?|between|difference|differ|than|"
                         r"compar\w*|entre|difference)\b")
_AGGREGATION = re.compile(r"\b(top|highest|lowest|most|least|rank\w*|average|mean|sum|total|across|all|every|"
                          r"overall|plus|moins|moyenne|tous|toutes|classe\w*)\b")
_FOLLOWUP = re.compile(r"\b(it|its|they|them|their|those|these|that one|same|also|and for|and what|what about|"
                       r"celui|celle|ceux|aussi|et pour|meme)\b")
_CLAUSE = re.compile(r"\b(and then|then|also|as well as|puis|ensuite)\b|[;?]\s*\S")


DEFAULT_FEATURES: tuple[Feature, ...] = (
    Feature("extra_cancers", 2.0, lambda i: max(0, len(i.mentions.cancers) + len(i.mentions.unknown_cancers) - 1)),
    Feature("genes_mentioned", 0.5, lambda i: min(len(i.mentions.genes), 4)),
    Feature("comparison", 2.0, lambda i: float(bool(_COMPARISON.search(normalize(i.question))))),
    Feature("aggregation_or_ranking", 1.0, lambda i: float(bool(_AGGREGATION.search(normalize(i.question))))),
    Feature("extra_clauses", 1.0, lambda i: len(_CLAUSE.findall(normalize(i.question)))),
    Feature("long_question", 1.0, lambda i: float(len(i.question.split()) > 25)),
    Feature("followup", 1.5, lambda i: float(i.has_history and bool(_FOLLOWUP.search(normalize(i.question))))),
)


class RuleBasedSelector:
    def __init__(self, threshold: float, features: tuple[Feature, ...] = DEFAULT_FEATURES):
        self.threshold = threshold
        self.features = features

    def select(self, inp: SelectionInput) -> Decision:
        values = {f.name: float(f.extract(inp)) for f in self.features}
        contributions = {f.name: values[f.name] * f.weight for f in self.features}
        score = round(sum(contributions.values()), 2)
        tier: Tier = "smart" if score >= self.threshold else "fast"
        top = [k for k, v in sorted(contributions.items(), key=lambda kv: -kv[1]) if v > 0][:3]
        reason = f"score {score} {'≥' if tier == 'smart' else '<'} {self.threshold}" + (
            f" ({', '.join(top)})" if top else "")
        return Decision(tier=tier, score=score, features=values, reason=reason)
