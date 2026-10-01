"""Deterministic entity resolution: the non-AI NLP layer.

Used in three places:
  * tools  -> map free-text tool arguments ("Breast cancer", "kidney", "brest") to catalog values;
  * router / model selector -> count entities to estimate complexity;
  * guards -> detect when the LLM silently substitutes a cancer (e.g. esophageal -> gastric).

Design note: tool arguments are free strings resolved here, NOT a JSON-schema enum. An enum would
force a small model to pick *some* listed value for "esophageal" (typically "gastric"), turning an honest
"not in the dataset" into a confident wrong answer.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

from app.data.repository import Catalog

# Synonyms -> catalog value. Deliberately NO entry for esophageal/oesophageal: it is not in the dataset and
# must not be mapped to gastric even though they are anatomically close.
SYNONYMS: dict[str, str] = {
    # English
    "breast": "breast", "mammary": "breast",
    "lung": "lung", "nsclc": "lung", "sclc": "lung", "pulmonary": "lung",
    "prostate": "prostate", "prostatic": "prostate",
    "gastric": "gastric", "stomach": "gastric",
    "glioblastoma": "glioblastoma", "gbm": "glioblastoma", "glioma": "glioblastoma", "brain": "glioblastoma",
    "colorectal": "colorectal", "colon": "colorectal", "rectal": "colorectal", "crc": "colorectal",
    "bowel": "colorectal",
    "melanoma": "melanoma", "skin": "melanoma",
    "ovarian": "ovarian", "ovary": "ovarian",
    "pancreatic": "pancreatic", "pancreas": "pancreatic", "pdac": "pancreatic",
    "renal": "renal", "kidney": "renal", "rcc": "renal",
    # French
    "sein": "breast", "mammaire": "breast",
    "poumon": "lung", "pulmonaire": "lung", "poumons": "lung",
    "prostatique": "prostate",
    "gastrique": "gastric", "estomac": "gastric",
    "cerveau": "glioblastoma",
    "rectum": "colorectal",  # "côlon"/"pancréas" normalise to the English keys
    "melanome": "melanoma", "peau": "melanoma",
    "ovaire": "ovarian", "ovaires": "ovarian", "ovarien": "ovarian",
    "pancreatique": "pancreatic",
    "rein": "renal", "reins": "renal", "renale": "renal",
}

_NOISE = r"\b(cancers?|tumou?rs?|carcinomas?|adenocarcinomas?|neoplasms?|du|de|la|le|l|des|d)\b"
# Words that may precede "cancer" without naming an indication.
_NOT_AN_INDICATION = {
    "the", "a", "an", "of", "in", "for", "with", "and", "or", "what", "which", "this", "that", "each", "every",
    "any", "all", "my", "your", "their", "other", "main", "same", "le", "la", "les", "un", "une", "du", "de", "des",
    "ce", "quel", "quels", "quelle", "quelles", "chaque", "tout", "tous", "autre", "types", "type", "kinds",
    "kind", "about", "is", "are", "available", "known", "given",
}
_EN_PATTERN = re.compile(r"\b([a-z][a-z\-]+)\s+(?:cancers?|tumou?rs?|carcinomas?)\b")
_FR_PATTERN = re.compile(r"\bcancers?\s+(?:du|de\s+la|de\s+l'|de\s+l|des|de|d')?\s*([a-z][a-z\-]+)")


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text.lower()).strip()


@dataclass
class Mentions:
    cancers: list[str] = field(default_factory=list)  # resolved catalog values, in order of appearance
    unknown_cancers: list[str] = field(default_factory=list)  # "X cancer" where X is not in the dataset
    genes: list[str] = field(default_factory=list)

    @property
    def n_entities(self) -> int:
        return len(self.cancers) + len(self.unknown_cancers) + len(self.genes)


class Resolver:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self._cancers = set(catalog.cancers)
        self._genes = set(catalog.genes)
        self._synonyms = {k: v for k, v in SYNONYMS.items() if v in self._cancers}
        for c in catalog.cancers:  # data-driven: a new indication in the CSV is resolvable without code change
            self._synonyms.setdefault(c, c)

    # --- cancers ---

    def resolve_cancer(self, text: str, fuzzy: bool = True) -> str | None:
        """Map free text to a catalog cancer value, or None if it is not in the dataset."""
        t = re.sub(_NOISE, " ", normalize(text).replace("'", " "))
        t = re.sub(r"[^a-z0-9 \-]", " ", t).strip()
        if not t:
            return None
        if t in self._synonyms:
            return self._synonyms[t]
        for word in t.split():
            if word in self._synonyms:
                return self._synonyms[word]
        if fuzzy:
            for word in [t, *t.split()]:
                if len(word) >= 4:
                    match = difflib.get_close_matches(word, list(self._synonyms), n=1, cutoff=0.85)
                    if match:
                        return self._synonyms[match[0]]
        return None

    def resolve_gene(self, text: str) -> str | None:
        g = text.strip().upper()
        return g if g in self._genes else None

    # --- free-text extraction ---

    def extract(self, question: str) -> Mentions:
        m = Mentions()
        q = normalize(question).replace("'", " ' ")
        words = re.findall(r"[a-z0-9\-]+", q)
        for w in words:
            c = self._synonyms.get(w)
            if c and c not in m.cancers:
                m.cancers.append(c)
        for pattern in (_EN_PATTERN, _FR_PATTERN):
            for cand in pattern.findall(q):
                if cand in _NOT_AN_INDICATION:
                    continue
                resolved = self.resolve_cancer(cand)
                if resolved is None:
                    if cand not in m.unknown_cancers:
                        m.unknown_cancers.append(cand)
                elif resolved not in m.cancers:
                    m.cancers.append(resolved)
        # Genes: short symbols (AR, MET, RET, ATM...) collide with English words -> require uppercase.
        for tok in re.findall(r"[A-Za-z0-9\-]+", question):
            g = tok.upper()
            if g in self._genes and (len(g) >= 4 or tok.isupper()) and g not in m.genes:
                m.genes.append(g)
        return m
