# Roadmap: anticipated evolutions

Status legend: **Done** = in this POC · **Ready** = the extension point exists, about a day of work ·
**Later** = only worth it at a larger scale.

## 1. More parameters, columns, tables or data

### 1a. ReAct agent and a real SQL database with a strong schema: **Ready**

- The agent is already a ReAct-style loop (reason, call a tool, observe, repeat; at most 4 iterations).
- All data access goes through `app/data/repository.py`. Moving to Postgres, DuckDB or SQLite means
  re-implementing `Repository`; tools, agent, guards and API don't change.
- When there are about 10 tables or more, add a **table-selection node** before `agent`. It could be
  deterministic (keyword or synonym match against a table catalog, the same technique as `Resolver`) or an LLM
  call on the `smart` tier only. Its output narrows the tools and schema sent to the agent, which keeps prompts
  short. Prompt length is the main latency driver on CPU (see ARCHITECTURE.md, Performance on CPU).
- Add a generic `run_readonly_sql` tool **only** behind allow-listed views, a read-only connection, a statement
  parser that accepts only `SELECT`, and row and time limits. Keep the typed tools for the frequent questions:
  they are faster, safer and testable.

### 1b. Schema cache at agent start: **Done**

`Repository` builds a `Catalog` (cancers, genes, row count, `data_version` hash, `schema_version`) once at
startup. It feeds the system prompt, the resolver lexicon, the help answer and the cache key. With a real
database, the catalog comes from `information_schema` and is refreshed when `data_version` changes.

### 1c. Streaming answers like a live chat: **Done**

`POST /api/chat/stream` sends Server-Sent Events: `step` (graph progress), `token` (LLM tokens), `reset` (an
escalation drops the draft), and `final` (the authoritative answer and tables).

## 2. Intent router: **Done** (rule-based)

The cases are help or greeting (deterministic, no LLM), data (agent with mandatory tools) and other (agent,
politely restates the scope).

Next step: when the rules are unsure (no domain keyword, no entity), call a tiny classifier. That could be the
fast model with a 1-token constrained output, or a logistic regression on embeddings trained from the
`requests` table, which already stores intent, outcome and feedback.

## 3. RAG over the schema (embedder, vector DB, retriever): **Later**

- **Not worth it now.** The schema fits in about 150 tokens of the system prompt. Retrieval would add an
  embedding model (RAM), a vector store, a retrieval step (latency), and a failure mode (missed retrieval) that
  doesn't exist today.
- **When it becomes worth it.** When the schema description exceeds the prompt budget, roughly 50+ tables or
  rich column documentation. Even then, the higher-value RAG is **few-shot retrieval of (question → tool plan)
  pairs**, mined from successful traces in `requests`, rather than schema chunks.
- **Retro-compatibility and versioning.** This is already prepared: every trace and cache key carries
  `data_version`, `schema_version` and `prompt_version`.
  - Embedded schema chunks would carry the same `schema_version` as metadata.
  - The retriever filters on the active version, and the old index stays until its traces are no longer
    replayed.
  - The eval harness is run per version before switching (blue/green index).
  - Prefer additive schema changes (new columns or tables) over renames. Renames need a mapping table in the
    resolver.

## 4. Restricting tables and forbidding raw SQL: **Done by design**

The LLM can only call the registered tools, and there is no query language to inject. Next steps when
confidential data appears:
- **Per-tool RBAC.** The tool list sent to the LLM depends on the user's role, so the model can't even see
  forbidden tools.
- **Row-level filters** inside `Repository`, applied after the LLM, so prompt injection can't bypass them.
- An **audit trail**, which already exists in `requests` and `steps`.

## 5. Other items worth considering

| Item | Status | Note |
|---|---|---|
| More primitives | Partly done | Done: `list_cancer_types`, `summarize_expressions`, `find_cancers_for_gene`, `top_genes`. Next: `compare_cancers(cancers, genes)` (deterministic comparison table), `genes_in_common(cancers)`, `expression_threshold(cancer, min_value)`, and gene annotation lookup (HGNC/Ensembl, cached offline). |
| User feedback | Ready | 👍/👎 on each answer, stored in `requests`. It turns traces into labelled eval data. |
| LLM-as-judge | Ready | Offline only (`eval/`), for answer faithfulness. Never on the request path: online checks stay deterministic and instant. |
| Semantic cache | Later | Only with an entity-equality guard: same resolved cancers and genes, and similarity of 0.95 or above. |
| Langfuse / OpenTelemetry | Ready | Export the `steps` rows as spans. Self-hosted Langfuse needs Postgres and ClickHouse, which is too heavy for the POC. |
| Auth and multi-user | Later | An API key header, then SSO. Conversations are already keyed by id. |
| Cloud LLM fallback | Later | The `OllamaFactory` interface takes any LangChain chat model, so a cloud model could become a third tier (`expert`) for questions that fail on `smart`, where policy allows. |
| Concurrency | Later | One Ollama slot is enough for a single-user POC. For a team, use vLLM or llama.cpp server with parallel slots and a request queue. |
