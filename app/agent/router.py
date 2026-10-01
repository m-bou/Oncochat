"""Rule-based intent router (no LLM call, ~0.1 ms).

help / greeting -> deterministic capability answer built from the tool registry and data catalog.
data            -> agent; tool use is mandatory (enforced by guards).
other           -> agent; may answer without tools (e.g. politely decline off-topic requests).

An LLM classifier can be added later as a fallback for low-confidence cases (docs/ROADMAP.md §2).
"""

import re
from dataclasses import dataclass
from typing import Literal

from app.agent.resolver import Mentions, normalize

Intent = Literal["help", "greeting", "data", "other"]

_HELP = re.compile(
    r"\b(how (can|could|do) you help|what (can|could) you do|what do you do|who are you|what are you|"
    r"help me|^help\b|capabilit|how does (this|it) work|"
    r"comment (peux|pouvez)[- ](tu|vous) m?.?aider|que (peux|pouvez)[- ](tu|vous) faire|aide[- ]moi|^aide\b|"
    r"qui es[- ]tu)"
)
_GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|bonjour|salut|bonsoir|coucou|"
                       r"thanks?( you)?|merci( beaucoup)?)\b")
_DOMAIN = re.compile(
    r"\b(genes?|genetic|targets?|expressions?|expressed|median|values?|cancers?|tumou?rs?|oncolog\w*|"
    r"indications?|mutations?|biomarkers?|genes?|cibles?|exprim\w*|mediane|valeurs?|onco\w*)\b"
)
_FRENCH = re.compile(
    r"\b(le|la|les|des|du|une?|est|sont|quels?|quelles?|quelle|comment|pourquoi|dans|avec|pour|"
    r"bonjour|salut|merci|peux|pouvez|aider|valeur|mediane)\b"
)
_ENGLISH = re.compile(r"\b(the|is|are|what|which|how|of|in|you|me|can|for|with|involved|main)\b")


@dataclass(frozen=True)
class Route:
    intent: Intent
    lang: Literal["en", "fr"]


def detect_language(question: str) -> Literal["en", "fr"]:
    """Stop-word vote; accented characters are a strong French signal."""
    q = normalize(question)
    fr_hits = len(_FRENCH.findall(q))
    en_hits = len(_ENGLISH.findall(q))
    return "fr" if fr_hits > en_hits or re.search(r"[éèêàçù]", question.lower()) else "en"


def route(question: str, mentions: Mentions) -> Route:
    q = normalize(question)
    lang = detect_language(question)
    if mentions.n_entities == 0 and _HELP.search(q):
        return Route("help", lang)
    if mentions.n_entities == 0 and _GREETING.match(q) and len(q.split()) <= 5:
        return Route("greeting", lang)
    if mentions.n_entities > 0 or _DOMAIN.search(q):
        return Route("data", lang)
    return Route("other", lang)
