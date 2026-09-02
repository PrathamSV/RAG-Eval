"""
Serving graph (production query path, hit on POST /query):

  query rewrite -> [route: exact error-code match, or vector retrieve]
    -- exact match path --
    code lookup -> [nodes found?] -> generate
                -> [no nodes found] -> retrieve (vector fallback)
    -- vector path --
    -> retrieve -> rerank -> sufficiency check
       -> [conditional: loop back to rewrite_query if context is weak, else]
       -> generate
    -> faithfulness self-check -> return answer + citations

Two corrective mechanisms live here, for two different failure modes:

  1. Exact error-code routing. If the question contains something
     code-shaped (e.g. "ABA0008", "aba-0008"), a metadata-filtered DB
     lookup runs FIRST, bypassing vector search entirely. This exists
     because embeddings are structurally bad at opaque alphanumeric IDs --
     two catalog rows for two different codes can share more surface
     language ("Critical", "Protection Policy", similar sentence shape)
     than either shares with its own code string, so the reranker/
     sufficiency-threshold machinery below can't reliably recover from a
     bad initial vector retrieval on these queries. A metadata equality
     match is deterministic, so this path skips rerank + sufficiency
     checking and goes straight to generate() once it finds anything.

  2. Corrective RAG (the original conditional edge): if the reranked top
     score is below threshold, the query gets rewritten and retried
     (bounded by MAX_RETRIEVE_ATTEMPTS) before the generation model is
     ever called. This still covers every query that ISN'T an exact code
     match -- descriptive questions, no-code queries, and code-shaped text
     that doesn't match anything in the catalog (typo'd digit, wrong
     version) and falls through to this path instead.
"""

import uuid
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

from common import configure_llamaindex_settings, fetch_nodes_by_error_code, get_llm, get_vector_store
from parsers import normalize_error_code

MAX_RETRIEVE_ATTEMPTS = 2
RETRIEVE_TOP_K = 10
RERANK_TOP_N = 4
SUFFICIENCY_THRESHOLD = 0.35  # min top reranked score to call context "enough"


class ServingState(TypedDict, total=False):
    request_id: str
    question: str
    rewritten_question: str
    attempts: int
    detected_code: Optional[str]
    code_lookup_attempted: bool
    retrieval_path: str  # "exact_match" | "vector" -- for citations/debugging
    retrieved_nodes: List[NodeWithScore]
    reranked_nodes: List[NodeWithScore]
    sufficient: bool
    answer: str
    faithfulness_score: float
    citations: List[dict]


_cached_index: VectorStoreIndex | None = None


def _build_index() -> VectorStoreIndex:
    global _cached_index
    if _cached_index is None:
        configure_llamaindex_settings()
        _cached_index = VectorStoreIndex.from_vector_store(get_vector_store())
    return _cached_index


_reranker_cache: dict[int, SentenceTransformerRerank] = {}


def _get_reranker(top_n: int) -> SentenceTransformerRerank:
    if top_n not in _reranker_cache:
        _reranker_cache[top_n] = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n
        )
    return _reranker_cache[top_n]

def rewrite_query(state: ServingState) -> dict:
    """First pass: use the question as-is, and check it for an exact error
    code while we're at it (see route_after_rewrite). On a retry (context
    judged insufficient on the vector path), broaden the phrasing so a
    second retrieval has a genuinely different shot at finding relevant
    passages -- code detection isn't repeated here, since the vector
    fallback only runs when a code either wasn't present or didn't match
    anything in the catalog, and that fact doesn't change on retry."""
    rid = state["request_id"]
    if state.get("attempts", 0) == 0:
        code = normalize_error_code(state["question"])
        if code:
            print(f"[{rid}] detected error code in question: {code}", flush=True)
        print(f"[{rid}] attempt 1: using question as-is", flush=True)
        return {"rewritten_question": state["question"], "attempts": 1, "detected_code": code}

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


def route_after_rewrite(state: ServingState) -> str:
    """Send exact-code-shaped questions to the deterministic lookup path
    first; everything else (no code, or a retry after the lookup already
    failed once) goes to normal vector retrieval."""
    rid = state["request_id"]
    if state.get("detected_code") and not state.get("code_lookup_attempted"):
        print(f"[{rid}] routing to exact-match code lookup", flush=True)
        return "code_lookup"
    return "retrieve"


def code_lookup(state: ServingState) -> dict:
    rid = state["request_id"]
    code = state["detected_code"]
    print(f"[{rid}] exact-match lookup for error_code={code}...", flush=True)
    nodes = fetch_nodes_by_error_code(code)
    print(f"[{rid}] exact-match lookup found {len(nodes)} chunk(s) for {code}", flush=True)
    # reranked_nodes (not retrieved_nodes) so this slots directly into
    # generate()/faithfulness_check() without needing a separate field --
    # an exact metadata match doesn't need reranking or a sufficiency
    # score, it's already the ground truth for this code.
    return {"reranked_nodes": nodes, "code_lookup_attempted": True, "retrieval_path": "exact_match"}


def route_after_code_lookup(state: ServingState) -> str:
    rid = state["request_id"]
    if state["reranked_nodes"]:
        print(f"[{rid}] exact match found -- skipping rerank/sufficiency, generating directly", flush=True)
        return "generate"
    print(
        f"[{rid}] '{state['detected_code']}' is code-shaped but matched no chunks in the "
        f"catalog -- falling back to vector retrieval",
        flush=True,
    )
    return "retrieve"


def retrieve(state: ServingState) -> dict:
    rid = state["request_id"]
    print(f"[{rid}] retrieving top-{RETRIEVE_TOP_K} chunks...", flush=True)
    index = _build_index()
    retriever = index.as_retriever(similarity_top_k=RETRIEVE_TOP_K)
    nodes = retriever.retrieve(state["rewritten_question"])
    print(f"[{rid}] retrieved {len(nodes)} chunk(s)", flush=True)
    return {"retrieved_nodes": nodes, "retrieval_path": "vector"}


def rerank(state: ServingState) -> dict:
    rid = state["request_id"]
    print(f"[{rid}] reranking to top-{RERANK_TOP_N}...", flush=True)
    reranker = _get_reranker(RERANK_TOP_N)
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
    graph.add_node("code_lookup", code_lookup)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("check_sufficiency", check_sufficiency)
    graph.add_node("generate", generate)
    graph.add_node("faithfulness_check", faithfulness_check)

    graph.set_entry_point("rewrite_query")
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"code_lookup": "code_lookup", "retrieve": "retrieve"},
    )
    graph.add_conditional_edges(
        "code_lookup",
        route_after_code_lookup,
        {"generate": "generate", "retrieve": "retrieve"},
    )
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