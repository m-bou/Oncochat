"""Data access layer: the ONLY place that touches the dataset.

Swapping the CSV for SQLite/Postgres/DuckDB later (docs/ROADMAP.md §1a) means re-implementing this class;
tools, agent and API stay untouched. The catalog built at startup is our "schema cache" (ROADMAP §1b).
"""

import hashlib
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCHEMA_VERSION = "1"  # bump when columns/semantics change; stamped in traces and cache keys


@dataclass(frozen=True)
class Catalog:
    cancers: tuple[str, ...]
    genes: tuple[str, ...]
    n_rows: int
    data_version: str
    schema_version: str = SCHEMA_VERSION

    def describe(self) -> str:
        """Compact schema description injected into the system prompt (~150 tokens)."""
        return (
            f"Table gene_expression ({self.n_rows} rows): cancer_indication (text), gene (text, HGNC symbol), "
            f"median_value (float, median expression, 0-1 scale). One row per (cancer_indication, gene). "
            f"Available cancer_indication values: {', '.join(self.cancers)}."
        )


class Repository:
    def __init__(self, csv_path: Path):
        raw = Path(csv_path).read_bytes()
        self.df = pd.read_csv(csv_path)
        self.df["cancer_indication"] = self.df["cancer_indication"].str.strip().str.lower()
        self.df["gene"] = self.df["gene"].str.strip().str.upper()
        self.catalog = Catalog(
            cancers=tuple(sorted(self.df["cancer_indication"].unique())),
            genes=tuple(sorted(self.df["gene"].unique())),
            n_rows=len(self.df),
            data_version=hashlib.sha256(raw).hexdigest()[:12],
        )

    # --- primitives (cancer names are expected to be already resolved to catalog values) ---

    def get_targets(self, cancer: str) -> list[str]:
        """Same semantics as the provided `get_targets` (CSV order preserved)."""
        return self.df[self.df["cancer_indication"] == cancer]["gene"].tolist()

    def get_expressions(self, genes: list[str], cancer: str | None = None) -> dict[str, float]:
        """Fixed version of the provided `get_expressions`: scoped to a cancer when given.

        Without `cancer`, a gene present in several indications is ambiguous; we then return one entry
        per (gene, cancer) as "GENE (cancer)" keys instead of silently keeping the last row.
        """
        subset = self.df[self.df["gene"].isin([g.upper() for g in genes])]
        if cancer is not None:
            subset = subset[subset["cancer_indication"] == cancer]
            return dict(zip(subset["gene"], subset["median_value"].astype(float), strict=True))
        counts = subset["gene"].value_counts()
        out: dict[str, float] = {}
        for gene, c, v in subset[["gene", "cancer_indication", "median_value"]].itertuples(index=False):
            out[gene if counts[gene] == 1 else f"{gene} ({c})"] = float(v)
        return out

    def summarize_expressions(self, cancer: str) -> dict:
        """Per-gene values sorted highest first, plus pre-computed facts.

        Small models misread unsorted lists ("TP53 is the lowest"), so rankings and extremes are stated
        explicitly instead of being left to the LLM to infer.
        """
        values = self.get_expressions(self.get_targets(cancer), cancer)
        ranked = sorted(values.items(), key=lambda kv: -kv[1])
        vals = [v for _, v in ranked]
        return {
            "cancer": cancer,
            "per_gene_highest_first": dict(ranked),
            "stats": {
                "n_genes": len(vals),
                "median_of_medians": round(statistics.median(vals), 4),
                "mean": round(statistics.fmean(vals), 4),
                "highest": {"gene": ranked[0][0], "median_value": ranked[0][1]},
                "lowest": {"gene": ranked[-1][0], "median_value": ranked[-1][1]},
            },
        }

    def find_cancers_for_gene(self, gene: str) -> dict[str, float]:
        subset = self.df[self.df["gene"] == gene.upper()].sort_values("median_value", ascending=False)
        return dict(zip(subset["cancer_indication"], subset["median_value"].astype(float), strict=True))

    def top_genes(self, cancer: str, n: int = 5, order: str = "desc") -> list[dict]:
        subset = self.df[self.df["cancer_indication"] == cancer]
        subset = subset.sort_values("median_value", ascending=(order == "asc")).head(n)
        return [{"gene": g, "median_value": float(v)} for g, v in zip(subset["gene"], subset["median_value"])]  # noqa: B905
