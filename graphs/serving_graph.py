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

from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI

from common import GENERATION_MODEL, configure_llamaindex_settings, get_vector_store

MAX_RETRIEVE_ATTEMPTS = 2
RETRIEVE_TOP_K = 10
RERANK_TOP_N = 4
SUFFICIENCY_THRESHOLD = 0.35  # min top reranked score to call context "enough"


class ServingState(TypedDict, total=False):
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
    if state.get("attempts", 0) == 0:
        return {"rewritten_question": state["question"], "attempts": 1}

    llm = GoogleGenAI(model=GENERATION_MODEL)
    prompt = (
        "Rewrite the following question to be broader and use different "
        "phrasing, so a vector search is more likely to find relevant "
        f"passages. Return ONLY the rewritten question.\n\nQuestion: {state['question']}"
    )
    rewritten = str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
    return {"rewritten_question": rewritten, "attempts": state["attempts"] + 1}


def retrieve(state: ServingState) -> dict:
    index = _build_index()
    retriever = index.as_retriever(similarity_top_k=RETRIEVE_TOP_K)
    nodes = retriever.retrieve(state["rewritten_question"])
    return {"retrieved_nodes": nodes}


def rerank(state: ServingState) -> dict:
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=RERANK_TOP_N
    )
    reranked = reranker.postprocess_nodes(
        state["retrieved_nodes"], query_str=state["rewritten_question"]
    )
    return {"reranked_nodes": reranked}


def check_sufficiency(state: ServingState) -> dict:
    nodes = state["reranked_nodes"]
    top_score = nodes[0].score if nodes and nodes[0].score is not None else 0.0
    return {"sufficient": top_score >= SUFFICIENCY_THRESHOLD}


def route_after_sufficiency(state: ServingState) -> str:
    if state["sufficient"] or state["attempts"] >= MAX_RETRIEVE_ATTEMPTS:
        return "generate"
    return "rewrite_query"


def generate(state: ServingState) -> dict:
    llm = GoogleGenAI(model=GENERATION_MODEL)
    context = "\n\n".join(n.node.get_content() for n in state["reranked_nodes"])
    prompt = (
        "Answer the question using ONLY the context below. If the context "
        "doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}\n\nAnswer:"
    )
    answer = str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
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

    context_chunks = [n.node.get_content() for n in state["reranked_nodes"]]
    score = faithfulness(state["answer"], context_chunks)
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
    return graph.invoke({"question": question, "attempts": 0})
