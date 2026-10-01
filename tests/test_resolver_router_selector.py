import pytest

from app.agent.model_selector import RuleBasedSelector, SelectionInput
from app.agent.router import detect_language, route

BRIEF = [
    "How can you help me?",
    "What are the main genes involved in lung cancer?",
    "What is the median value expression of genes involved in breast cancer?",
    "What is the median value expression of genes involved in esophageal cancer?",
]


@pytest.mark.parametrize("text,expected", [
    ("lung", "lung"), ("Breast cancer", "breast"), ("kidney", "renal"), ("stomach", "gastric"),
    ("NSCLC", "lung"), ("brest", "breast"), ("pancreatic cancer", "pancreatic"), ("cancer du sein", "breast"),
    ("mélanome", "melanoma"), ("esophageal", None), ("oesophageal cancer", None), ("liver", None), ("", None),
])
def test_resolve_cancer(resolver, text, expected):
    assert resolver.resolve_cancer(text) == expected


def test_extract_mentions(resolver):
    m = resolver.extract(BRIEF[3])
    assert m.cancers == [] and m.unknown_cancers == ["esophageal"]
    m = resolver.extract("Compare KRAS in lung and pancreatic cancer")
    assert m.cancers == ["lung", "pancreatic"] and m.genes == ["KRAS"]
    m = resolver.extract("Quels sont les gènes du cancer du sein ?")
    assert m.cancers == ["breast"] and m.unknown_cancers == []
    # short gene symbols must be uppercase to count (avoid "met", "at"...)
    assert resolver.extract("we met at the AR meeting").genes == ["AR"]
    assert resolver.extract("main genes in lung cancer").unknown_cancers == []


@pytest.mark.parametrize("q,intent", [
    (BRIEF[0], "help"), ("Comment peux-tu m'aider ?", "help"), ("hello", "greeting"), ("Bonjour !", "greeting"),
    (BRIEF[1], "data"), (BRIEF[3], "data"), ("TP53?", "data"), ("What's the weather in Paris?", "other"),
])
def test_route(resolver, q, intent):
    assert route(q, resolver.extract(q)).intent == intent


def test_language():
    assert detect_language(BRIEF[1]) == "en"
    assert detect_language("Quels sont les principaux gènes impliqués dans le cancer du poumon ?") == "fr"
    assert detect_language("Quelle est la valeur mediane pour le sein") == "fr"


def test_selector_brief_queries_use_fast_tier(resolver):
    sel = RuleBasedSelector(threshold=3.0)
    for q in BRIEF:
        d = sel.select(SelectionInput(q, resolver.extract(q), has_history=False))
        assert d.tier == "fast", (q, d)


def test_selector_complex_query_uses_smart_tier(resolver):
    sel = RuleBasedSelector(threshold=3.0)
    q = "Compare the expression of KRAS between lung, pancreatic and colorectal cancer"
    d = sel.select(SelectionInput(q, resolver.extract(q), has_history=False))
    assert d.tier == "smart" and d.features["extra_cancers"] == 2 and d.features["comparison"] == 1
    assert "extra_cancers" in d.reason
