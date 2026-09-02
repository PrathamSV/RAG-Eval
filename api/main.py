"""
FastAPI service boundary: ingest, query (serving graph), eval-set
generation, eval runs + comparison, and feedback capture.

Run: uvicorn api.main:app --reload
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
from eval.generate_testset import main as generate_testset_main
from graphs.eval_graph import run_eval
from graphs.serving_graph import run_serving_graph
from eval.metrics import bootstrap_ci, correlation, paired_significance_test

app = FastAPI(title="RAG MVP")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list
    faithfulness_score: float
    attempts: int


class FeedbackRequest(BaseModel):
    query_text: str
    answer_text: str
    rating: int  # 1 (thumbs up) or -1 (thumbs down)
    retrieved_chunk_ids: Optional[list] = None


class GenerateTestsetRequest(BaseModel):
    n: int = 20
    multi_hop_fraction: float = 0.5


class EvalRunRequest(BaseModel):
    use_reranker: bool = True
    rerank_top_n: int = 4
    top_k: int = 10
    # When True, also writes a JSON file with each query's question,
    # reference answer, generated answer, gold/retrieved chunk ids, and
    # every metric -- for manual inspection alongside the DB-backed run.
    save_details: bool = False
    # Optional explicit path for that file. Defaults to
    # ./eval_exports/eval_run_<id>_<timestamp>.json when omitted. Ignored
    # if save_details is False.
    output_path: Optional[str] = None


@app.on_event("startup")
def startup() -> None:
    db.init_schema()
    from common import ensure_fulltext_index
    ensure_fulltext_index()


@app.post("/admin/truncate")
def truncate_all(include_chunks: bool = True) -> dict:
    """Wipes eval_queries/eval_runs/eval_results/feedback (and, by default,
    the ingested pgvector chunks in data_rag_chunks too) and resets their id
    counters. Pass ?include_chunks=false to keep the ingested corpus and
    only clear eval/feedback data. Irreversible — there's no confirmation
    step, so treat this as a dev/reset tool rather than something wired
    into a UI button without a guard in front of it."""
    truncated = db.truncate_all(include_chunks=include_chunks)
    return {"status": "ok", "truncated": truncated}


@app.post("/ingest")
def ingest() -> dict:
    """Triggers the same ingestion pipeline as `python ingest.py`, reading
    from ./data. The CLI script stays the source of truth for ingestion
    logic; this is a thin wrapper so it's also reachable over HTTP."""
    import ingest as ingest_module

    ingest_module.main()
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    result = run_serving_graph(req.question)
    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        faithfulness_score=result["faithfulness_score"],
        attempts=result["attempts"],
    )


@app.post("/eval/generate-testset")
def generate_testset(req: GenerateTestsetRequest) -> dict:
    generate_testset_main(req.n, req.multi_hop_fraction)
    return {"status": "ok", "requested": req.n}


@app.post("/eval/run")
def eval_run(req: EvalRunRequest) -> dict:
    try:
        result = run_eval(
            use_reranker=req.use_reranker,
            rerank_top_n=req.rerank_top_n,
            top_k=req.top_k,
            save_details=req.save_details,
            output_path=req.output_path,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"run_id": result["run_id"], "details_path": result["details_path"]}


@app.get("/eval/runs/{run_id}")
def get_run(run_id: int) -> dict:
    run, agg = db.fetch_run_summary(run_id)
    if run is None:
        raise HTTPException(404, "run not found")

    metrics_by_query = db.fetch_run_metrics_by_query(run_id)
    metric_names = [
        "hit_rate", "recall", "precision", "mrr", "latency_ms",
        "faithfulness", "answer_relevance", "answer_correctness",
    ]
    confidence_intervals = {
        m: bootstrap_ci([row.get(m) for row in metrics_by_query.values()])
        for m in metric_names
    }
    return {"run": run, "metrics": agg, "confidence_intervals": confidence_intervals}


@app.get("/eval/runs/{run_id}/details")
def get_run_details(run_id: int) -> list:
    return db.fetch_run_details(run_id)


@app.get("/eval/compare")
def compare_runs(run_a: int, run_b: int) -> dict:
    """Diffs two eval runs -- e.g. reranker off (run_a) vs on (run_b) -- so
    a component's value shows up as a metric delta with a paired
    significance test attached, instead of a bare number you have to take
    on faith with ~20 queries."""
    _, agg_a = db.fetch_run_summary(run_a)
    _, agg_b = db.fetch_run_summary(run_b)
    if agg_a is None or agg_b is None:
        raise HTTPException(404, "one or both runs not found")

    metrics_by_query_a = db.fetch_run_metrics_by_query(run_a)
    metrics_by_query_b = db.fetch_run_metrics_by_query(run_b)
    common_query_ids = sorted(set(metrics_by_query_a) & set(metrics_by_query_b))

    metric_names = [
        "hit_rate", "recall", "precision", "mrr", "latency_ms",
        "faithfulness", "answer_relevance", "answer_correctness",
    ]

    delta = {}
    significance = {}
    for key in metric_names:
        va, vb = agg_a.get(f"avg_{key}"), agg_b.get(f"avg_{key}")
        delta[key] = (vb - va) if (va is not None and vb is not None) else None

        values_a = [metrics_by_query_a[qid].get(key) for qid in common_query_ids]
        values_b = [metrics_by_query_b[qid].get(key) for qid in common_query_ids]
        significance[key] = paired_significance_test(values_a, values_b)

    return {
        "run_a": run_a,
        "run_b": run_b,
        "metrics_a": agg_a,
        "metrics_b": agg_b,
        "delta": delta,
        "significance": significance,
        "n_common_queries": len(common_query_ids),
    }


@app.get("/eval/runs/{run_id}/correlations")
def get_run_correlations(run_id: int) -> dict:
    """Correlates each retrieval metric (hit_rate/recall/precision/mrr)
    against each generation metric (faithfulness/answer_relevance/
    answer_correctness) across the run's queries -- checks whether better
    retrieval actually predicts better generation for this run/corpus,
    rather than assuming precision@k-style metrics matter just because
    they're computed (see TODO.md's note that precision@k can be
    misleading in isolation)."""
    metrics_by_query = db.fetch_run_metrics_by_query(run_id)
    if not metrics_by_query:
        raise HTTPException(404, "run not found or has no results")

    retrieval_metric_names = ["hit_rate", "recall", "precision", "mrr"]
    generation_metric_names = ["faithfulness", "answer_relevance", "answer_correctness"]
    rows = list(metrics_by_query.values())

    matrix = {}
    for r_metric in retrieval_metric_names:
        r_values = [row.get(r_metric) for row in rows]
        matrix[r_metric] = {
            g_metric: correlation(r_values, [row.get(g_metric) for row in rows])
            for g_metric in generation_metric_names
        }

    return {"run_id": run_id, "n_queries": len(rows), "correlations": matrix}


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    if req.rating not in (1, -1):
        raise HTTPException(400, "rating must be 1 or -1")
    db.insert_feedback(
        query_text=req.query_text,
        answer_text=req.answer_text,
        rating=req.rating,
        retrieved_chunk_ids=req.retrieved_chunk_ids,
    )
    return {"status": "ok"}