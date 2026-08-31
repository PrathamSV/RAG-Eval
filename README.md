# RAG MVP — ingestion, retrieval, generation, orchestration, eval

A full RAG stack: LlamaIndex handles chunking/embedding/retrieval, Postgres +
pgvector stores the vectors *and* the eval tables, LangGraph orchestrates two
distinct workflows (a corrective-RAG serving graph and a fan-out/fan-in eval
graph), and FastAPI is the service boundary over all of it. Gemini (free
tier, via Google AI Studio) provides the embedding model, and Anthropic for both the
generation and judge LLMs.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

docker compose up -d
psql -h localhost -U postgres -d ragdb -c "CREATE EXTENSION IF NOT EXISTS vector;"

cp .env.example .env   # then add your GOOGLE_API_KEY (free from https://aistudio.google.com/apikey)

python ingest.py                          # loads ./data into pgvector
python query.py                           # phase-1 sanity check: ask a question, get a grounded answer + sources

# bring up the full service
uvicorn api.main:app --reload
```


## Building an eval set and running eval

```bash
python -m eval.generate_testset --n 20   # or: POST /eval/generate-testset {"n": 20}
python -c "from graphs.eval_graph import run_eval; print(run_eval(use_reranker=False))"
python -c "from graphs.eval_graph import run_eval; print(run_eval(use_reranker=True))"
```

```
GET /eval/compare?run_a=<no-reranker-run-id>&run_b=<reranker-run-id>
```

## Architecture

| Tool | Role |
|---|---|
| **LlamaIndex** | Document loaders, chunker/node parser, embedding pipeline, retriever, and the `SentenceTransformerRerank` cross-encoder postprocessor. |
| **pgvector** | Postgres extension storing chunk embeddings (`data_rag_chunks`, managed by LlamaIndex's `PGVectorStore`). The *same* database also holds `eval_queries`, `eval_runs`, `eval_results`, and `feedback` — one system instead of a vector DB plus a separate metrics DB. |
| **LangGraph** | Two graphs: the **serving graph** (production query pipeline with a real conditional loop) and the **eval graph** (a per-query fan-out/fan-in DAG, driven over the whole eval set). |
| **FastAPI** | `/ingest`, `/query`, `/eval/generate-testset`, `/eval/run`, `/eval/runs/{id}`, `/eval/runs/{id}/details`, `/eval/compare`, `/feedback`. |

### Data model (Postgres)

- `data_rag_chunks` — created automatically by LlamaIndex's `PGVectorStore`; holds chunk text, `embedding vector(768)`, and metadata (including `node_id`, which every other table references as its "chunk id").
- `eval_queries` — query text, `gold_chunk_ids text[]` (node_ids from `data_rag_chunks`), optional `reference_answer`, `query_type` (single-hop / multi-hop).
- `eval_runs` — run id, timestamps, `config` snapshot (embedding model, reranker on/off, top_k) — what makes before/after comparisons possible.
- `eval_results` — per-query, per-run scores for every metric (`hit_rate`, `recall`, `precision`, `mrr`, `faithfulness`, `answer_relevance`, `answer_correctness`, `latency_ms`).
- `feedback` — thumbs up/down from production traffic, for online eval.

### Serving graph (`graphs/serving_graph.py`) [WIP]

```
rewrite_query → retrieve → rerank → check_sufficiency
   ├─(context weak, attempts left)→ rewrite_query   (loop)
   └─(sufficient, or out of attempts)→ generate → faithfulness_check → END
```

The conditional edge is genuine corrective RAG: if the reranked top score is
below `SUFFICIENCY_THRESHOLD`, the question gets rewritten and retried
(bounded by `MAX_RETRIEVE_ATTEMPTS`) before the generation model is ever
called. Every response includes citations (node id, score, snippet) and a
self-reported faithfulness score.

### Eval graph (`graphs/eval_graph.py`) [WIP]

Per query:

```
retrieve ─┬→ compute_retrieval_metrics ─┐
          └→ generate → compute_generation_metrics ─┴→ combine → END
```

Retrieval metrics (Hit Rate@K, Recall@K, Precision@K, MRR — pure
list-comparison, no LLM call) run off `retrieve` in parallel with generation
and its own LLM-judge metrics (faithfulness, answer relevance, answer
correctness); `combine` fans back in once both branches finish. `run_eval()`
is the outer "load eval set" step: it loops this compiled graph over every
row in `eval_queries`, persists each result, and finalizes the `eval_runs`
row (`status='complete'`/`'failed'`).

## Files

- `common.py` — shared LlamaIndex Settings (LLM + embedding model) and Postgres connection helpers, imported by every other file so ingestion and querying can never drift apart.
- `ingest.py` — chunks and embeds everything in `./data`, writes to `data_rag_chunks`.
- `query.py` — phase-1 CLI script: embeds a question, retrieves top-k, prints an answer + sources. Superseded in production by the serving graph.
- `db.py` — psycopg2 helpers for `eval_queries` / `eval_runs` / `eval_results` / `feedback` (schema in `schema.sql`).
- `schema.sql` — DDL for the eval + feedback tables.
- `eval/metrics.py` — Hit Rate@K, Recall@K, Precision@K, MRR (pure functions) plus LLM-judge faithfulness / answer relevance / answer correctness.
- `eval/generate_testset.py` — samples ingested chunks, prompts the LLM for a question + reference answer per chunk, writes `eval_queries` rows. Also `POST /eval/generate-testset`.
- `graphs/serving_graph.py` — the corrective-RAG production pipeline described above. Also `POST /query`.
- `graphs/eval_graph.py` — the fan-out/fan-in eval pipeline described above, plus `run_eval()` which drives it over the whole eval set. Also `POST /eval/run`.
- `api/main.py` — FastAPI app wiring all of the above into HTTP endpoints, plus `GET /eval/compare` for before/after diffing and `POST /feedback`.
- `data/sample_doc.txt` — starter document so ingestion works out of the box. Drop in your own `.txt`/`.pdf`/`.md` files and re-run `ingest.py` (then regenerate the eval set, since gold chunk ids are tied to specific ingested chunks).
