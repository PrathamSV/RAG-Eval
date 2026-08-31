"""
Serving graph (production query path, hit on POST /query):

  query rewrite -> retrieve -> rerank -> sufficiency check
    -> [conditional: loop back to retrieve if context is weak, else]
    -> generate -> faithfulness self-check -> return answer + citations

The conditional edge is what makes this "corrective RAG" rather than a
straight pipeline: if the reranked top score is below threshold, the query
gets rewritten and retried (bounded by MAX_RETRIEVE_ATTEMPTS) before we ever
call the generation model.
"""

import uuid
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

from common import configure_llamaindex_settings, get_llm, get_vector_store

MAX_RETRIEVE_ATTEMPTS = 2
RETRIEVE_TOP_K = 10
RERANK_TOP_N = 4
SUFFICIENCY_THRESHOLD = 0.35  # min top reranked score to call context "enough"


class ServingState(TypedDict, total=False):
    request_id: str
    question: str
    rewritten_question: str
    attempts: int
    retrieved_nodes: List[NodeWithScore]
    reranked_nodes: List[NodeWithScore]
    sufficient: bool
    answer: str
    faithfulness_score: float
    citations: List[dict]


def _build_index() -> VectorStoreIndex:
    configure_llamaindex_settings()
    return VectorStoreIndex.from_vector_store(get_vector_store())


def rewrite_query(state: ServingState) -> dict:
    """First pass: use the question as-is. On a retry (context judged
    insufficient), broaden the phrasing so a second retrieval has a
    genuinely different shot at finding relevant passages."""
    rid = state["request_id"]
    if state.get("attempts", 0) == 0:
        print(f"[{rid}] attempt 1: using question as-is", flush=True)
        return {"rewritten_question": state["question"], "attempts": 1}

    print(f"[{rid}] attempt {state['attempts'] + 1}: rewriting question for retry...", flush=True)
    llm = get_llm()
    prompt = (
        "Rewrite the following question to be broader and use different "
        "phrasing, so a vector search is more likely to find relevant "
        f"passages. Return ONLY the rewritten question.\n\nQuestion: {state['question']}"
    )
    rewritten = str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
    print(f"[{rid}] rewritten question: {rewritten!r}", flush=True)
    return {"rewritten_question": rewritten, "attempts": state["attempts"] + 1}


def retrieve(state: ServingState) -> dict:
    rid = state["request_id"]
    print(f"[{rid}] retrieving top-{RETRIEVE_TOP_K} chunks...", flush=True)
    index = _build_index()
    retriever = index.as_retriever(similarity_top_k=RETRIEVE_TOP_K)
    nodes = retriever.retrieve(state["rewritten_question"])
    print(f"[{rid}] retrieved {len(nodes)} chunk(s)", flush=True)
    return {"retrieved_nodes": nodes}


def rerank(state: ServingState) -> dict:
    rid = state["request_id"]
    print(f"[{rid}] reranking to top-{RERANK_TOP_N}...", flush=True)
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=RERANK_TOP_N
    )
    reranked = reranker.postprocess_nodes(
        state["retrieved_nodes"], query_str=state["rewritten_question"]
    )
    print(f"[{rid}] reranked to {len(reranked)} chunk(s)", flush=True)
    return {"reranked_nodes": reranked}


def check_sufficiency(state: ServingState) -> dict:
    rid = state["request_id"]
    nodes = state["reranked_nodes"]
    top_score = nodes[0].score if nodes and nodes[0].score is not None else 0.0
    sufficient = top_score >= SUFFICIENCY_THRESHOLD
    print(
        f"[{rid}] sufficiency check: top score {top_score:.3f} "
        f"(threshold {SUFFICIENCY_THRESHOLD}) -> {'sufficient' if sufficient else 'insufficient'}",
        flush=True,
    )
    return {"sufficient": sufficient}


def route_after_sufficiency(state: ServingState) -> str:
    rid = state["request_id"]
    if state["sufficient"] or state["attempts"] >= MAX_RETRIEVE_ATTEMPTS:
        print(f"[{rid}] proceeding to generate", flush=True)
        return "generate"
    print(f"[{rid}] context insufficient, retrying retrieval", flush=True)
    return "rewrite_query"


def generate(state: ServingState) -> dict:
    rid = state["request_id"]
    print(f"[{rid}] generating answer...", flush=True)
    llm = get_llm()
    context = "\n\n".join(n.node.get_content() for n in state["reranked_nodes"])
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"
    )
    answer = str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
    print(f"[{rid}] answer generated ({len(answer)} chars)", flush=True)
    citations = [
        {
            "node_id": n.node.node_id,
            "score": n.score,
            "snippet": n.node.get_content()[:200],
        }
        for n in state["reranked_nodes"]
    ]
    return {"answer": answer, "citations": citations}


def faithfulness_check(state: ServingState) -> dict:
    from eval.metrics import faithfulness

    rid = state["request_id"]
    print(f"[{rid}] running faithfulness self-check...", flush=True)
    context_chunks = [n.node.get_content() for n in state["reranked_nodes"]]
    score = faithfulness(state["answer"], context_chunks)
    print(f"[{rid}] faithfulness score: {score} — done", flush=True)
    return {"faithfulness_score": score}


def build_serving_graph():
    graph = StateGraph(ServingState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("check_sufficiency", check_sufficiency)
    graph.add_node("generate", generate)
    graph.add_node("faithfulness_check", faithfulness_check)

    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency",
        route_after_sufficiency,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    graph.add_edge("generate", "faithfulness_check")
    graph.add_edge("faithfulness_check", END)

    return graph.compile()


_serving_graph = None


def get_serving_graph():
    global _serving_graph
    if _serving_graph is None:
        _serving_graph = build_serving_graph()
    return _serving_graph


def run_serving_graph(question: str) -> ServingState:
    graph = get_serving_graph()
    request_id = uuid.uuid4().hex[:8]
    print(f"[{request_id}] received question: {question[:80]!r}", flush=True)
    try:
        result = graph.invoke({"question": question, "attempts": 0, "request_id": request_id})
    except Exception as exc:
        print(f"[{request_id}] FAILED: {exc!r}", flush=True)
        raise
    print(f"[{request_id}] request complete", flush=True)
    return result