"""
Eval graph (offline, hit on POST /eval/run):

  load eval set
    -> for each query: retrieve (exact error-code match first, vector
       search as fallback -- see retrieve() below)
      -> [branch] compute retrieval metrics IN PARALLEL WITH run generation -> compute generation metrics
      -> [join] combine
    -> persist to eval_runs / eval_results

The fan-out/fan-in (retrieval metrics computed independently of, and
concurrently with, generation + its own metrics) is expressed as an actual
LangGraph branch — `compute_retrieval_metrics` and `generate` both read off
`retrieve`, and `combine` waits on both — rather than a linear script.
`run_eval` is the "load eval set" step: it loops the compiled per-query graph
over every row in eval_queries and persists as it goes.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

import db
from common import (
    GENERATION_MODEL,
    configure_llamaindex_settings,
    fetch_nodes_by_error_code,
    get_llm,
    get_vector_store,
)
from eval.metrics import generation_metrics, retrieval_metrics
from parsers import normalize_error_code

RETRIEVE_TOP_K = 10

# Where per-query export files land when save_details=True and no explicit
# output_path is given. Kept out of the repo tables entirely -- this is a
# side file for manual eyeballing, not something downstream code reads.
DEFAULT_EXPORT_DIR = Path("./eval_exports")


class EvalQueryState(TypedDict, total=False):
    query_id: int
    question: str
    gold_chunk_ids: List[str]
    reference_answer: Optional[str]
    use_reranker: bool
    rerank_top_n: int
    top_k: int
    detected_code: Optional[str]
    retrieval_path: str  # "exact_match" | "vector"
    retrieved_nodes: List[NodeWithScore]
    retrieved_ids: List[str]
    latency_ms: float
    retrieval_scores: dict
    answer: str
    generation_scores: dict
    combined: dict


_cached_index: VectorStoreIndex | None = None


def _index() -> VectorStoreIndex:
    """Build the index once and cache it at module level -- retrieve() is
    called once per query in run_eval()'s loop, and rebuilding
    VectorStoreIndex from scratch each time reconstructs the embedding
    model (HuggingFaceEmbedding), which reloads its weights off disk on
    every call. Caching means the weights load once per process instead
    of once per query."""
    global _cached_index
    if _cached_index is None:
        configure_llamaindex_settings()
        _cached_index = VectorStoreIndex.from_vector_store(get_vector_store())
    return _cached_index


_reranker_cache: dict[int, SentenceTransformerRerank] = {}


def _get_reranker(top_n: int) -> SentenceTransformerRerank:
    """Cache SentenceTransformerRerank instances by top_n -- constructing
    one loads the cross-encoder weights from disk, and retrieve() runs
    once per query in run_eval()'s loop, so building a fresh instance per
    query reloads the weights every time (same issue as the embedding
    model before it was cached in _index())."""
    if top_n not in _reranker_cache:
        _reranker_cache[top_n] = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n
        )
    return _reranker_cache[top_n]


def retrieve(state: EvalQueryState) -> dict:
    """Exact error-code match first (deterministic metadata lookup,
    bypassing both vector search and the reranker -- see
    fetch_nodes_by_error_code's docstring for why embeddings are
    unreliable on opaque alphanumeric codes specifically). Only falls
    through to normal vector retrieval when no code is detected, or the
    detected code is well-formed but not actually in the ingested catalog.
    """
    qid = state["query_id"]
    start = time.perf_counter()

    code = normalize_error_code(state["question"])
    if code:
        print(f"  [query {qid}] detected error code {code} -- exact-match lookup", flush=True)
        nodes = fetch_nodes_by_error_code(code)
        if nodes:
            latency_ms = (time.perf_counter() - start) * 1000
            ids = [n.node.node_id for n in nodes]
            print(
                f"  [query {qid}] exact-match lookup found {len(ids)} chunk(s) for {code} "
                f"in {latency_ms:.0f}ms",
                flush=True,
            )
            return {
                "retrieved_nodes": nodes,
                "retrieved_ids": ids,
                "latency_ms": latency_ms,
                "detected_code": code,
                "retrieval_path": "exact_match",
            }
        print(
            f"  [query {qid}] '{code}' is code-shaped but matched no chunks in the catalog "
            f"-- falling back to vector search",
            flush=True,
        )

    top_k = state.get("top_k", RETRIEVE_TOP_K)
    print(f"  [query {qid}] retrieving top-{top_k} chunks...", flush=True)
    index = _index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(state["question"])

    if state.get("use_reranker"):
        rerank_top_n = state.get("rerank_top_n", 4)
        print(f"  [query {qid}] reranking to top-{rerank_top_n}...", flush=True)
        reranker = _get_reranker(rerank_top_n)
        nodes = reranker.postprocess_nodes(nodes, query_str=state["question"])

    latency_ms = (time.perf_counter() - start) * 1000
    ids = [n.node.node_id for n in nodes]
    print(f"  [query {qid}] retrieved {len(ids)} chunk(s) in {latency_ms:.0f}ms", flush=True)
    return {
        "retrieved_nodes": nodes,
        "retrieved_ids": ids,
        "latency_ms": latency_ms,
        "detected_code": code,
        "retrieval_path": "vector",
    }


def compute_retrieval_metrics(state: EvalQueryState) -> dict:
    scores = retrieval_metrics(state["retrieved_ids"], state["gold_chunk_ids"])
    print(f"  [query {state['query_id']}] retrieval metrics: {scores}", flush=True)
    return {"retrieval_scores": scores}


def generate(state: EvalQueryState) -> dict:
    qid = state["query_id"]
    print(f"  [query {qid}] generating answer ({GENERATION_MODEL})...", flush=True)
    llm = get_llm()
    context = "\n\n".join(n.node.get_content() for n in state["retrieved_nodes"])
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"
    )
    answer = str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
    print(f"  [query {qid}] answer generated ({len(answer)} chars)", flush=True)
    return {"answer": answer}


def compute_generation_metrics(state: EvalQueryState) -> dict:
    qid = state["query_id"]
    print(f"  [query {qid}] scoring faithfulness / relevance / correctness...", flush=True)
    context_chunks = [n.node.get_content() for n in state["retrieved_nodes"]]
    scores = generation_metrics(
        state["question"], state["answer"], context_chunks, state.get("reference_answer")
    )
    print(f"  [query {qid}] generation metrics: {scores}", flush=True)
    return {"generation_scores": scores}


def combine(state: EvalQueryState) -> dict:
    return {"combined": {**state["retrieval_scores"], **state["generation_scores"]}}


def build_eval_query_graph():
    graph = StateGraph(EvalQueryState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("compute_retrieval_metrics", compute_retrieval_metrics)
    graph.add_node("generate", generate)
    graph.add_node("compute_generation_metrics", compute_generation_metrics)
    # defer=True: the retrieval-metrics branch is 1 hop (retrieve ->
    # compute_retrieval_metrics) while the generation branch is 2 hops
    # (retrieve -> generate -> compute_generation_metrics). Without `defer`,
    # LangGraph fires `combine` as soon as the SHORT branch finishes, before
    # generation_scores exists -> KeyError. `defer=True` holds this node
    # until every pending branch in the step has actually completed.
    graph.add_node("combine", combine, defer=True)

    graph.set_entry_point("retrieve")
    # fan-out: retrieval metrics and generation both branch off `retrieve`
    graph.add_edge("retrieve", "compute_retrieval_metrics")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "compute_generation_metrics")
    # fan-in: combine only runs once both branches have finished
    graph.add_edge("compute_retrieval_metrics", "combine")
    graph.add_edge("compute_generation_metrics", "combine")
    graph.add_edge("combine", END)

    return graph.compile()


_eval_query_graph = None


def get_eval_query_graph():
    global _eval_query_graph
    if _eval_query_graph is None:
        _eval_query_graph = build_eval_query_graph()
    return _eval_query_graph


def _default_export_path(run_id: int) -> Path:
    """`./eval_exports/eval_run_<id>_<timestamp>.json` -- timestamped so
    re-running the same run_id (or running eval repeatedly while iterating)
    never silently clobbers a previous export."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_EXPORT_DIR / f"eval_run_{run_id}_{stamp}.json"


def _write_details_file(
    output_path: Path, run_id: int, config: dict, records: List[dict]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "config": config,
        "n_queries": len(records),
        "queries": records,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_eval(
    use_reranker: bool = True,
    rerank_top_n: int = 4,
    top_k: int = RETRIEVE_TOP_K,
    save_details: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """Load the eval set, run every query through the per-query graph, persist
    per-query results, and finalize the eval_runs row. Returns
    {"run_id": ..., "details_path": <str or None>}.

    save_details: if True, also writes a JSON file with, per query, the
        question, reference (gold) answer, generated answer, gold/retrieved
        chunk ids, which retrieval path served it (exact_match vs vector),
        and every metric (hit_rate/recall/precision/mrr/latency_ms/
        faithfulness/answer_relevance/answer_correctness) -- meant purely
        for manual inspection alongside the DB-backed run, not read back by
        any other code path.
    output_path: where to write that file. Defaults to
        `./eval_exports/eval_run_<id>_<timestamp>.json` if omitted. Ignored
        if save_details is False.
    """
    if use_reranker and rerank_top_n > top_k:
        raise ValueError(
            f"rerank_top_n ({rerank_top_n}) cannot exceed top_k ({top_k}) -- "
            "the reranker only reorders/trims the top_k candidates already "
            "retrieved, so asking it to keep more than that doesn't make sense."
        )

    db.init_schema()
    queries = db.fetch_eval_queries()
    if not queries:
        raise SystemExit("No eval queries found — run eval/generate_testset.py first.")

    config = {
        "embedding_model": "gemini-embedding-001",
        "generation_model": GENERATION_MODEL,
        "use_reranker": use_reranker,
        "rerank_top_n": rerank_top_n,
        "top_k": top_k,
    }
    run_id = db.create_eval_run(config)
    graph = get_eval_query_graph()

    print(
        f"Starting eval run {run_id} over {len(queries)} quer"
        f"{'y' if len(queries) == 1 else 'ies'} "
        f"(reranker={'on' if use_reranker else 'off'}, top_k={top_k})...",
        flush=True,
    )

    # Only accumulated in memory when save_details is on -- an eval run can
    # be large, so there's no reason to hold every question/answer/context
    # for the lifetime of the run if nobody asked for the export.
    detail_records: List[dict] = [] if save_details else None

    try:
        for i, q in enumerate(queries, start=1):
            print(
                f"[{i}/{len(queries)}] query {q['id']}: {q['query_text'][:80]!r}",
                flush=True,
            )
            try:
                result = graph.invoke(
                    {
                        "query_id": q["id"],
                        "question": q["query_text"],
                        "gold_chunk_ids": q["gold_chunk_ids"],
                        "reference_answer": q["reference_answer"],
                        "use_reranker": use_reranker,
                        "rerank_top_n": rerank_top_n,
                        "top_k": top_k,
                    }
                )
                db.insert_eval_result(
                    run_id=run_id,
                    query_id=q["id"],
                    retrieved_chunk_ids=result["retrieved_ids"],
                    latency_ms=result["latency_ms"],
                    metrics=result["combined"],
                    generated_answer=result["answer"],
                )
            except Exception as exc:
                print(f"[{i}/{len(queries)}] FAILED on query {q['id']}: {exc!r}", flush=True)
                raise
            print(f"[{i}/{len(queries)}] ok — persisted", flush=True)

            if save_details:
                detail_records.append(
                    {
                        "query_id": q["id"],
                        "query_type": q["query_type"],
                        "question": q["query_text"],
                        "reference_answer": q["reference_answer"],
                        "generated_answer": result["answer"],
                        "retrieval_path": result.get("retrieval_path"),
                        "detected_code": result.get("detected_code"),
                        "gold_chunk_ids": q["gold_chunk_ids"],
                        "retrieved_chunk_ids": result["retrieved_ids"],
                        "latency_ms": result["latency_ms"],
                        "metrics": result["combined"],
                    }
                )

        db.finish_eval_run(run_id, status="complete")
        print(f"Eval run {run_id} complete — {len(queries)} quer"
              f"{'y' if len(queries) == 1 else 'ies'} evaluated.", flush=True)
    except Exception:
        db.finish_eval_run(run_id, status="failed")
        print(f"Eval run {run_id} marked failed.", flush=True)
        raise

    details_path = None
    if save_details:
        path = Path(output_path) if output_path else _default_export_path(run_id)
        _write_details_file(path, run_id, config, detail_records)
        details_path = str(path)
        print(f"Eval run {run_id} details written to {details_path}", flush=True)

    return {"run_id": run_id, "details_path": details_path}