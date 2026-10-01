"""The two functions provided with the exercise, kept VERBATIM (only the CSV path is parametrised).

Do not "fix" this file: it is the reference implementation the brief asks us to orchestrate.

Known defect (pinned by tests/test_primitives.py):
`get_expressions` filters on gene only. Genes shared by several cancers (TP53, KRAS, BRCA2, ...) have one
row per cancer, and `dict(zip(...))` keeps the LAST row of the CSV. For breast cancer, 5 out of 10 values
returned are values from another indication. The agent uses the cancer-scoped version in
`app/data/repository.py` instead.
"""

from typing import Dict, List  # noqa: UP035 - verbatim copy

import pandas as pd

from app.config import get_settings

# Load CSV
df = pd.read_csv(get_settings().data_path)


def get_targets(cancer_name: str) -> List[str]:  # noqa: UP006
    """Return a list of genes for a given cancer type."""
    return df[df["cancer_indication"] == cancer_name]["gene"].tolist()


def get_expressions(genes: List[str]) -> Dict[str, float]:  # noqa: UP006
    """Return the median values for the given list of genes."""
    subset = df[df["gene"].isin(genes)]
    return dict(zip(subset["gene"], subset["median_value"]))  # noqa: B905
