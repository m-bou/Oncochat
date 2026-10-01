from app.agent import guards
from app.agent.tools import to_table


def test_tool_unknown_cancer_is_structured(toolbox):
    r = toolbox.execute("get_targets", {"cancer": "esophageal"})
    assert r["error"] == "unknown_cancer" and "gastric" in r["available"]


def test_tool_lenient_args(toolbox):
    r = toolbox.execute("get_expressions", {"genes": "TP53, KRAS", "cancer": "Lung cancer"})
    assert r["cancer"] == "lung" and r["expressions"] == {"KRAS": 0.359}
    assert r["not_measured_in_this_cancer"] == ["TP53"]


def test_tool_compare_cancers(toolbox):
    r = toolbox.execute("get_expressions", {"genes": ["KRAS"], "cancer": ["lung", "colorectal"]})
    assert r["expressions_by_cancer"] == {"lung": {"KRAS": 0.359}, "colorectal": {"KRAS": 0.885}}
    tables = to_table("get_expressions", {}, r)
    assert tables[0]["rows"] == [["lung", 0.359], ["colorectal", 0.885]]
    assert toolbox.execute("get_expressions", {"genes": ["KRAS"], "cancer": ["lung", "liver"]})["error"] == (
        "unknown_cancer")


def test_tool_misuse(toolbox):
    assert toolbox.execute("drop_table", {})["error"] == "unknown_tool"
    wrong = toolbox.execute("find_cancers_for_gene", {"gene": "renal"})
    assert wrong["error"] == "wrong_tool" and "get_targets(cancer='renal')" in wrong["message"]
    assert toolbox.execute("top_genes", {"cancer": "lung", "n": 0})["error"] == "invalid_arguments"


def test_schemas_are_openai_style(toolbox):
    names = {s["function"]["name"] for s in toolbox.schemas()}
    assert {"get_targets", "get_expressions", "summarize_expressions", "list_cancer_types"} <= names


def test_to_table(toolbox):
    r = toolbox.execute("summarize_expressions", {"cancer": "breast"})
    t = to_table("summarize_expressions", {}, r)
    assert t["columns"] == ["gene", "median_value"] and len(t["rows"]) == 10 and "lowest BRCA2" in t["footer"]
    assert to_table("get_targets", {}, {"error": "unknown_cancer"}) is None


def _results(toolbox, name, **args):
    return [{"name": name, "args": args, "result": toolbox.execute(name, args)}]


def test_grounding(toolbox):
    res = _results(toolbox, "summarize_expressions", cancer="breast")
    assert guards.ungrounded_numbers("TP53 is 0.233, ESR1 is the highest (0.72).", res) == []
    assert guards.ungrounded_numbers("TP53 vaut 0,233 soit 23.3 %", res) == []
    assert guards.ungrounded_numbers("TP53 is 0.373", res) == ["0.373"]
    # median of medians is 0.4345: both half-up (0.435) and half-even (0.434) roundings are grounded
    assert guards.ungrounded_numbers("The median is 0.435 (0.43).", res) == []


def test_top_genes_order_is_explicit(toolbox):
    r = toolbox.execute("top_genes", {"cancer": "prostate", "n": 3, "order": "highest"})
    assert [g["gene"] for g in r["genes"]] == ["MYC", "CDK12", "ATM"]
    assert toolbox.execute("top_genes", {"cancer": "prostate", "order": "asc"})["error"] == "invalid_arguments"


def check(resolver, q, answer, results, intent="data"):
    return guards.check(answer=answer, intent=intent, mentions=resolver.extract(q), tool_results=results,
                        resolver=resolver, hit_max_iterations=False)


def test_guard_ok_unknown_cancer(resolver, toolbox):
    q = "What is the median value expression of genes involved in esophageal cancer?"
    g = check(resolver, q, "Esophageal cancer is not in the dataset.", _results(toolbox, "get_targets",
                                                                                cancer="esophageal"))
    assert g.ok, g.reasons


def test_guard_catches_substitution(resolver, toolbox):
    q = "What is the median value expression of genes involved in esophageal cancer?"
    g = check(resolver, q, "Here are the values.", _results(toolbox, "summarize_expressions", cancer="gastric"))
    assert not g.ok and g.reasons[0].startswith("cancer_substitution")


def test_guard_unknown_cancer_answered_with_other_data(resolver, toolbox):
    """Regression (found manually): after a breast turn, the 3B model answered the esophageal question by querying
    ESR1 globally, a substitution through a gene rather than a cancer."""
    q = "What is the median value expression of genes involved in esophageal cancer?"
    g = check(resolver, q, "ESR1 in esophageal cancer is 0.716.", _results(toolbox, "find_cancers_for_gene",
                                                                        gene="ESR1"))
    assert not g.ok
    assert any(r.startswith("unknown_cancer_not_checked") for r in g.reasons)
    assert any(r.startswith("data_for_unknown_cancer") for r in g.reasons)
    # an honest answer needs no tool call: the catalog is authoritative
    assert check(resolver, q, "Esophageal cancer is not in the dataset. Available: breast, lung.", []).ok
    assert not check(resolver, q, "Esophageal is not available, but ESR1 is highly expressed.", []).ok
    # a gene explicitly asked about may be reported (e.g. "Is TP53 involved in esophageal cancer?")
    q2 = "Is TP53 involved in esophageal cancer?"
    res = _results(toolbox, "get_targets", cancer="esophageal") + _results(toolbox, "find_cancers_for_gene",
                                                                           gene="TP53")
    assert check(resolver, q2, "Esophageal is not in the dataset; TP53 is highest in ovarian (0.972).", res).ok


def test_guard_requires_tool_for_data(resolver):
    g = check(resolver, "What are the main genes in lung cancer?", "ALK and KRAS.", [])
    assert not g.ok and "no_tool_called_for_data_question" in g.reasons
    assert check(resolver, "What's the weather?", "I can only help with the dataset.", [], intent="other").ok


def test_guard_llm_error(resolver):
    g = guards.check(answer="", intent="data", mentions=resolver.extract("x"), tool_results=[],
                     resolver=resolver, hit_max_iterations=False, llm_error="ConnectError: refused")
    assert not g.ok and g.reasons[0].startswith("llm_error")
