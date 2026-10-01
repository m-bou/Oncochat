"""Tool registry: typed wrappers exposing the data primitives to the LLM.

Every tool:
  * validates its arguments with Pydantic (lenient coercion for small models, e.g. "TP53, KRAS" -> list);
  * resolves free-text entities through the Resolver;
  * returns a JSON-serialisable dict; failures are *structured* ({"error": ..., "available": [...]}) so the
    LLM can explain them instead of guessing.

No tool executes arbitrary code or queries: the attack surface is the list below.
Adding a primitive = one args model + one function + one ToolSpec entry.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.agent.resolver import Resolver
from app.data.repository import Repository


class NoArgs(BaseModel):
    pass


class CancerArgs(BaseModel):
    cancer: str = Field(description="Cancer indication in plain words, exactly as the user said it, e.g. 'lung'")


class ExpressionArgs(BaseModel):
    genes: list[str] = Field(description="Gene symbols, e.g. ['TP53', 'KRAS']")
    cancer: str | list[str] | None = Field(
        default=None,
        description="Cancer indication to scope the values to, or a list of indications to compare. Strongly "
        "recommended: the same gene has a different value in each cancer.",
    )

    @field_validator("genes", mode="before")
    @classmethod
    def _split(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [g.strip() for g in v.replace(";", ",").split(",") if g.strip()]
        return v


class GeneArgs(BaseModel):
    gene: str = Field(description="Gene symbol, e.g. 'TP53'")


class TopGenesArgs(BaseModel):
    cancer: str = Field(description="Cancer indication, e.g. 'prostate'")
    n: int = Field(default=5, ge=1, le=50, description="Number of genes to return")
    order: Literal["highest", "lowest"] = Field(
        default="highest", description="'highest' = most expressed genes first, 'lowest' = least expressed first")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[..., dict]
    example: str  # shown in the "How can you help me?" answer

    def schema(self) -> dict:
        params = self.args.model_json_schema()
        params.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}


class Toolbox:
    def __init__(self, repo: Repository, resolver: Resolver):
        self.repo = repo
        self.resolver = resolver
        self.specs: dict[str, ToolSpec] = {s.name: s for s in self._build()}

    # ------------------------------------------------------------------ helpers
    def _cancer(self, text: str) -> tuple[str | None, dict | None]:
        c = self.resolver.resolve_cancer(text)
        if c is None:
            return None, {"error": "unknown_cancer", "input": text,
                          "message": f"'{text}' is not a cancer indication available in the dataset.",
                          "available": list(self.repo.catalog.cancers)}
        return c, None

    # ------------------------------------------------------------------ tools
    def list_cancer_types(self) -> dict:
        return {"cancers": list(self.repo.catalog.cancers)}

    def get_targets(self, cancer: str) -> dict:
        c, err = self._cancer(cancer)
        genes = [] if err else self.repo.get_targets(c)
        return err or {"cancer": c, "n_genes": len(genes), "genes": genes}

    def get_expressions(self, genes: list[str], cancer: str | list[str] | None = None) -> dict:
        known = [g for g in (self.resolver.resolve_gene(x) for x in genes) if g]
        unknown = [x for x in genes if not self.resolver.resolve_gene(x)]
        if isinstance(cancer, list) and len(cancer) == 1:
            cancer = cancer[0]
        if isinstance(cancer, list):  # comparison across indications
            by_cancer: dict[str, dict[str, float]] = {}
            for name in cancer:
                c, err = self._cancer(name)
                if err:
                    return err
                by_cancer[c] = self.repo.get_expressions(known, c)
            return {"genes": known, "expressions_by_cancer": by_cancer, "unknown_genes": unknown}
        if cancer is None:
            return {"cancer": None, "expressions": self.repo.get_expressions(known), "unknown_genes": unknown}
        c, err = self._cancer(cancer)
        if err:
            return err
        values = self.repo.get_expressions(known, c)
        return {"cancer": c, "expressions": values, "unknown_genes": unknown,
                "not_measured_in_this_cancer": [g for g in known if g not in values]}

    def summarize_expressions(self, cancer: str) -> dict:
        c, err = self._cancer(cancer)
        return err or self.repo.summarize_expressions(c)

    def find_cancers_for_gene(self, gene: str) -> dict:
        g = self.resolver.resolve_gene(gene)
        if g is None and (c := self.resolver.resolve_cancer(gene, fuzzy=False)):
            return {"error": "wrong_tool", "input": gene,
                    "message": f"'{gene}' is a cancer indication, not a gene. Use get_targets(cancer='{c}') "
                               f"or summarize_expressions(cancer='{c}')."}
        if g is None:
            return {"error": "unknown_gene", "input": gene, "message": f"Gene '{gene}' is not in the dataset."}
        cancers = self.repo.find_cancers_for_gene(g)
        return {"gene": g, "n_cancers": len(cancers), "cancers_highest_first": cancers}

    def top_genes(self, cancer: str, n: int = 5, order: str = "highest") -> dict:
        c, err = self._cancer(cancer)
        return err or {"cancer": c, "order": order,
                       "genes": self.repo.top_genes(c, n, "asc" if order == "lowest" else "desc")}

    def _build(self) -> list[ToolSpec]:
        return [
            ToolSpec("list_cancer_types", "List the cancer indications available in the dataset.", NoArgs,
                     self.list_cancer_types, "Which cancers do you have data on?"),
            ToolSpec("get_targets", "Return the genes (targets) involved in a given cancer indication.",
                     CancerArgs, self.get_targets, "What are the main genes involved in lung cancer?"),
            ToolSpec("get_expressions",
                     "Return the median expression value of the given genes, scoped to a cancer indication.",
                     ExpressionArgs, self.get_expressions, "What is the expression of TP53 and KRAS in lung?"),
            ToolSpec("summarize_expressions",
                     "For a cancer indication, return the median expression value of every gene involved plus "
                     "summary statistics (median of medians, mean, min, max). Use it for any question about "
                     "expression values of the genes of a cancer.",
                     CancerArgs, self.summarize_expressions,
                     "What is the median value expression of genes involved in breast cancer?"),
            ToolSpec("find_cancers_for_gene",
                     "Return every cancer indication in which a gene is involved, with its median expression.",
                     GeneArgs, self.find_cancers_for_gene, "In which cancers is TP53 involved?"),
            ToolSpec("top_genes", "Rank the genes of a cancer indication by median expression.",
                     TopGenesArgs, self.top_genes, "What are the 3 most expressed genes in prostate cancer?"),
        ]

    # ------------------------------------------------------------------ execution
    def schemas(self) -> list[dict]:
        return [s.schema() for s in self.specs.values()]

    def execute(self, name: str, args: dict | None) -> dict:
        spec = self.specs.get(name)
        if spec is None:
            return {"error": "unknown_tool", "input": name, "available": list(self.specs)}
        try:
            parsed = spec.args.model_validate(args or {})
        except ValidationError as e:
            return {"error": "invalid_arguments", "message": e.errors(include_url=False, include_input=False)}
        return spec.fn(**parsed.model_dump())


# Errors that mean "the model used the tools wrongly" (guard + escalation) vs. legitimate negative answers.
MISUSE_ERRORS = {"unknown_tool", "invalid_arguments", "wrong_tool"}


def to_table(name: str, args: dict, result: dict) -> dict | list[dict] | None:
    """Turn a tool result into table(s) the UI renders directly (numbers never pass through the LLM)."""
    if "error" in result:
        return None
    if name == "get_expressions" and "expressions_by_cancer" in result:
        by = result["expressions_by_cancer"]
        return [{"title": f"{gene} median expression by cancer", "columns": ["cancer_indication", "median_value"],
                 "rows": [[c, vals[gene]] for c, vals in by.items() if gene in vals]} for gene in result["genes"]]
    if name == "get_targets":
        return {"title": f"Genes involved in {result['cancer']} cancer", "columns": ["gene"],
                "rows": [[g] for g in result["genes"]]}
    if name == "get_expressions":
        scope = f" in {result['cancer']} cancer" if result.get("cancer") else ""
        return {"title": f"Median expression{scope}", "columns": ["gene", "median_value"],
                "rows": [[g, v] for g, v in result["expressions"].items()]}
    if name == "summarize_expressions":
        s = result["stats"]
        return {"title": f"Median expression of genes involved in {result['cancer']} cancer",
                "columns": ["gene", "median_value"],
                "rows": [[g, v] for g, v in result["per_gene_highest_first"].items()],
                "footer": f"{s['n_genes']} genes. Median across genes {s['median_of_medians']}, mean {s['mean']}, "
                          f"highest {s['highest']['gene']} ({s['highest']['median_value']}), "
                          f"lowest {s['lowest']['gene']} ({s['lowest']['median_value']})."}
    if name == "find_cancers_for_gene":
        return {"title": f"Cancers involving {result['gene']}", "columns": ["cancer_indication", "median_value"],
                "rows": [[c, v] for c, v in result["cancers_highest_first"].items()]}
    if name == "top_genes":
        return {"title": f"{result['order'].capitalize()} expressed genes in {result['cancer']} cancer",
                "columns": ["gene", "median_value"],
                "rows": [[r["gene"], r["median_value"]] for r in result["genes"]]}
    return None
