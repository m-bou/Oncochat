import pytest

from app.data import primitives_original as original

BREAST_TRUTH = {"BRCA2": 0.032, "BRCA1": 0.094, "TP53": 0.233, "GATA3": 0.602, "CDH1": 0.561, "ESR1": 0.716,
                "MAP3K1": 0.701, "HER2": 0.42, "PIK3CA": 0.449, "AKT1": 0.278}


def test_original_get_targets_lung():
    assert original.get_targets("lung") == ["ALK", "RET", "ROS1", "STK11", "KRAS"]


def test_original_get_expressions_is_wrong_for_shared_genes():
    """Regression test documenting the defect of the provided function (see primitives_original.py)."""
    values = original.get_expressions(original.get_targets("breast"))
    wrong = {g for g, v in values.items() if v != BREAST_TRUTH[g]}
    assert wrong == {"BRCA2", "BRCA1", "TP53", "CDH1", "PIK3CA"}
    assert values["TP53"] == 0.373  # renal value, last TP53 row of the CSV


def test_scoped_get_expressions_is_correct(repo):
    assert repo.get_expressions(repo.get_targets("breast"), "breast") == BREAST_TRUTH


def test_unscoped_get_expressions_disambiguates(repo):
    values = repo.get_expressions(["TP53", "GATA3"])
    assert values["GATA3"] == 0.602
    assert values["TP53 (breast)"] == 0.233 and values["TP53 (renal)"] == 0.373


def test_summarize(repo):
    s = repo.summarize_expressions("breast")
    assert s["per_gene_highest_first"] == BREAST_TRUTH
    assert list(s["per_gene_highest_first"])[:2] == ["ESR1", "MAP3K1"]
    assert s["stats"]["n_genes"] == 10
    assert s["stats"]["median_of_medians"] == pytest.approx((0.42 + 0.449) / 2)
    assert s["stats"]["lowest"] == {"gene": "BRCA2", "median_value": 0.032}
    assert s["stats"]["highest"] == {"gene": "ESR1", "median_value": 0.716}


def test_find_cancers_and_top_genes(repo):
    assert repo.find_cancers_for_gene("tp53")["breast"] == 0.233
    assert len(repo.find_cancers_for_gene("TP53")) == 8
    assert [g["gene"] for g in repo.top_genes("prostate", 2)] == ["MYC", "CDK12"]
    assert repo.top_genes("ovarian", 1, "asc")[0] == {"gene": "KRAS", "median_value": 0.003}


def test_catalog(repo):
    c = repo.catalog
    assert len(c.cancers) == 10 and "esophageal" not in c.cancers
    assert c.n_rows == 81 and len(c.data_version) == 12
