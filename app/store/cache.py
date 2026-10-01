"""Exact-match answer cache key.

Not a semantic cache: "median expression in breast cancer" and "... in esophageal cancer" are near-identical
embeddings but need opposite answers. The key includes every input that can change the answer, so editing
the CSV, a prompt or a model invalidates stale entries automatically.
"""

import hashlib
import re
import unicodedata


def normalize_question(q: str) -> str:
    q = unicodedata.normalize("NFKD", q).encode("ascii", "ignore").decode().lower()
    q = re.sub(r"[^\w\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def cache_key(question: str, *, data_version: str, prompt_version: str, models: tuple[str, ...]) -> str:
    material = "|".join([normalize_question(question), data_version, prompt_version, *models])
    return hashlib.sha256(material.encode()).hexdigest()
