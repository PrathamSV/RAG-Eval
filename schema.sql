-- Eval + feedback tables. The chunks themselves live in the pgvector-managed
-- table `data_rag_chunks` (created automatically by llama-index's
-- PGVectorStore -- see ingest.py / common.py::get_vector_store). We don't
-- duplicate a separate `documents`/`chunks` table; eval_queries.gold_chunk_ids
-- stores the `node_id` values from data_rag_chunks directly, so retrieval
-- results (also node_ids) can be compared straightforwardly.

CREATE TABLE IF NOT EXISTS eval_queries (
    id               SERIAL PRIMARY KEY,
    query_text       TEXT NOT NULL,
    gold_chunk_ids   TEXT[] NOT NULL,
    reference_answer TEXT,
    query_type       TEXT NOT NULL DEFAULT 'single-hop',  -- 'single-hop' | 'multi-hop'
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id           SERIAL PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    config       JSONB NOT NULL,                 -- embedding model, top_k, reranker on/off, etc.
    status       TEXT NOT NULL DEFAULT 'running'  -- 'running' | 'complete' | 'failed'
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                  SERIAL PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    query_id            INTEGER NOT NULL REFERENCES eval_queries(id) ON DELETE CASCADE,
    retrieved_chunk_ids TEXT[] NOT NULL,
    latency_ms          DOUBLE PRECISION,
    hit_rate            DOUBLE PRECISION,
    recall              DOUBLE PRECISION,
    precision           DOUBLE PRECISION,
    mrr                 DOUBLE PRECISION,
    generated_answer    TEXT,
    faithfulness        DOUBLE PRECISION,
    answer_relevance    DOUBLE PRECISION,
    answer_correctness  DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
    id                  SERIAL PRIMARY KEY,
    query_text          TEXT NOT NULL,
    answer_text         TEXT NOT NULL,
    rating              SMALLINT NOT NULL CHECK (rating IN (-1, 1)),  -- thumbs down / up
    retrieved_chunk_ids TEXT[],
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_eval_results_run_id ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_query_id ON eval_results(query_id);
