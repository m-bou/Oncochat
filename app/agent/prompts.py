"""Prompts and deterministic answer templates. Bump PROMPT_VERSION on any change (it is part of the cache key
and stamped in every trace, so evaluations can be compared across prompt versions)."""

from app.agent.resolver import Mentions
from app.agent.tools import ToolSpec
from app.data.repository import Catalog

PROMPT_VERSION = "2026-10-01.6"

SYSTEM_PROMPT = """You are OncoChat, an assistant that answers questions about a small gene-expression dataset \
for non-technical stakeholders.

DATA: {schema}

RULES
1. Answer ONLY from tool results. Never use outside knowledge for genes, cancers or values.
2. For any question about genes, targets or expression values, call a tool first.
3. Pass cancer names to tools exactly as the user wrote them (e.g. "esophageal"). Never replace a cancer the \
user asked about by another one, even a similar one.
4. If a tool returns an error such as unknown_cancer, say clearly that this cancer is not in the dataset and \
list the available indications. Do not guess.
5. Copy numbers exactly as returned by tools. Do not compute statistics or rankings yourself: only state \
"highest", "lowest" or an order when the tool result says so explicitly.
6. Be concise: at most 3 short sentences. The user interface already shows the full tables returned by tools, \
so do not list every value: give the count and the key facts stated by the tool. All genes returned for a \
cancer are equally "involved"; never split them into more or less important ones.
7. If the question is unrelated to this dataset, say politely what you can help with instead.
8. Answer in the language requested in the user message."""

# Appended to the LAST user message, never to the system prompt: the system prompt + tool schemas must stay
# byte-identical across requests so Ollama reuses its KV cache for that ~1000-token prefix
# (docs/ARCHITECTURE.md, "Prompt layout").
HINT_TEMPLATE = (
    "\n\n(Parser notes, not from the user: cancers mentioned that are in the dataset: {cancers}; "
    "cancers mentioned that are NOT in the dataset: {unknown}; genes mentioned: {genes}. Answer in {language}.)"
)

FALLBACK = {
    "en": "I don't know — I could not find a reliable answer to this in the dataset. "
          "I can answer questions about these cancer indications: {cancers}.",
    "fr": "Je ne sais pas — je n'ai pas trouvé de réponse fiable dans le jeu de données. "
          "Je peux répondre sur ces indications : {cancers}.",
}
LLM_DOWN = {
    "en": "The language model is unavailable right now ({error}). Please check that Ollama is running "
          "(GET /api/health).",
    "fr": "Le modèle de langage est indisponible ({error}). Vérifiez qu'Ollama est démarré (GET /api/health).",
}


def system_prompt(catalog: Catalog) -> str:
    return SYSTEM_PROMPT.format(schema=catalog.describe())


DATA_REMINDER = " This is a data question: call a tool first, even if an earlier answer seems related."

# Guard reason code -> instruction given to the model on the retry (escalation with critique).
CRITIQUE = {
    "no_tool_called_for_data_question": "You did not call any tool. Call the appropriate tool to get the data.",
    "ungrounded_numbers": "Some numbers you wrote are not in any tool result ({detail}). Only use numbers "
                          "returned by tools.",
    "cancer_substitution": "You queried a cancer the user did not ask about ({detail}). Use exactly the cancer the "
                           "user mentioned, even if it is not in the dataset.",
    "tool_misuse": "Your tool calls had invalid arguments. Check the tool parameters and call again.",
    "unknown_cancer_not_checked": "The user asked about {detail}. Call a tool with exactly that cancer name to "
                                  "check whether it is in the dataset.",
    "data_for_unknown_cancer": "The user asked about {detail}, which is not in the dataset. Do not give values "
                               "from other cancers or earlier answers; say it is not available.",
    "max_tool_iterations_reached": "Stop calling tools and answer with the results you already have.",
    "empty_answer": "Your answer was empty. Answer the question.",
}
RETRY_TEMPLATE = "Your previous answer was rejected by an automatic check:\n{items}\nPlease try again."


def hint(m: Mentions, lang: str, intent: str = "data") -> str:
    def fmt(x: list[str]) -> str:
        return ", ".join(x) if x else "none"

    text = HINT_TEMPLATE.format(cancers=fmt(m.cancers), unknown=fmt(m.unknown_cancers), genes=fmt(m.genes),
                                language="French" if lang == "fr" else "English")
    return text[:-1] + DATA_REMINDER + ")" if intent == "data" else text


def critique(reasons: list[str]) -> str:
    items = []
    for r in reasons:
        code, _, detail = r.partition(": ")
        if code in CRITIQUE:
            items.append("- " + CRITIQUE[code].format(detail=detail))
    return RETRY_TEMPLATE.format(items="\n".join(items))


def help_answer(lang: str, catalog: Catalog, specs: list[ToolSpec], greeting: bool = False) -> str:
    cancers = ", ".join(catalog.cancers)
    examples = "\n".join(f"- *{s.example}*" for s in specs)
    if lang == "fr":
        head = "Bonjour ! " if greeting else ""
        return (f"{head}Je réponds en langage naturel à des questions sur un jeu de données d'expression génique "
                f"({catalog.n_rows} lignes) couvrant {len(catalog.cancers)} indications : {cancers}.\n\n"
                f"Je peux lister les gènes impliqués dans un cancer, donner leurs valeurs médianes "
                f"d'expression, classer les gènes ou retrouver les cancers associés à un gène. "
                f"Exemples (en anglais ou en français) :\n{examples}\n\n"
                f"Je réponds uniquement à partir des données : si une information n'y est pas, je le dis.")
    head = "Hello! " if greeting else ""
    return (f"{head}I answer natural-language questions about a gene-expression dataset ({catalog.n_rows} rows) "
            f"covering {len(catalog.cancers)} cancer indications: {cancers}.\n\n"
            f"I can list the genes involved in a cancer, give their median expression values, rank genes, or "
            f"find the cancers a gene is involved in. For example:\n{examples}\n\n"
            f"I only answer from the data: if something is not in it, I will tell you.")
