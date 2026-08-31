"""
Eval graph (offline, hit on POST /eval/run):

  load eval set
    -> for each query: retrieve
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

import time
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI

import db
from common import GENERATION_MODEL, configure_llamaindex_settings, get_vector_store
from eval.metrics import generation_metrics, retrieval_metrics

RETRIEVE_TOP_K = 10


class EvalQueryState(TypedDict, total=False):
    query_id: int
    question: str
    gold_chunk_ids: List[str]
    reference_answer: Optional[str]
    use_reranker: bool
    rerank_top_n: int
    retrieved_nodes: List[NodeWithScore]
    retrieved_ids: List[str]
    latency_ms: float
    retrieval_scores: dict
    answer: str
    generation_scores: dict
    combined: dict


def _index() -> VectorStoreIndex:
    configure_llamaindex_settings()
    return VectorStoreIndex.from_vector_store(get_vector_store())


def retrieve(state: EvalQueryState) -> dict:
    start = time.perf_counter()
    index = _index()
    retriever = index.as_retriever(similarity_top_k=RETRIEVE_TOP_K)
    nodes = retriever.retrieve(state["question"])

    if state.get("use_reranker"):
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=state.get("rerank_top_n", 4),
        )
        nodes = reranker.postprocess_nodes(nodes, query_str=state["question"])

    latency_ms = (time.perf_counter() - start) * 1000
    ids = [n.node.node_id for n in nodes]
    return {"retrieved_nodes": nodes, "retrieved_ids": ids, "latency_ms": latency_ms}


def compute_retrieval_metrics(state: EvalQueryState) -> dict:
    scores = retrieval_metrics(state["retrieved_ids"], state["gold_chunk_ids"])
    return {"retrieval_scores": scores}


def generate(state: EvalQueryState) -> dict:
    llm = GoogleGenAI(model=GENERATION_MODEL)
    context = "\n\n".join(n.node.get_content() for n in state["retrieved_nodes"])
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"
    )
    answer = str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
    return {"answer": answer}


def compute_generation_metrics(state: EvalQueryState) -> dict:
    context_chunks = [n.node.get_content() for n in state["retrieved_nodes"]]
    scores = generation_metrics(
        state["question"], state["answer"], context_chunks, state.get("reference_answer")
    )
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


def run_eval(use_reranker: bool = True, rerank_top_n: int = 4, top_k: int = RETRIEVE_TOP_K) -> int:
    """Load the eval set, run every query through the per-query graph, persist
    per-query results, and finalize the eval_runs row. Returns the run id."""
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

    try:
        for q in queries:
            result = graph.invoke(
                {
                    "query_id": q["id"],
                    "question": q["query_text"],
                    "gold_chunk_ids": q["gold_chunk_ids"],
                    "reference_answer": q["reference_answer"],
                    "use_reranker": use_reranker,
                    "rerank_top_n": rerank_top_n,
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
        db.finish_eval_run(run_id, status="complete")
    except Exception:
        db.finish_eval_run(run_id, status="failed")
        raise

    return run_id
