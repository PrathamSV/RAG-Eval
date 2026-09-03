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
from typing import Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore

from common import (
    EXPANSION_CHAR_CAP,
    WINDOWED_EXPANSION_TOP_M,
    GENERATE_PROMPT,
    build_numbered_context,
    expand_sop_context,
    fetch_nodes_by_error_code,
    get_cached_index,
    get_llm,
    get_reranker,
    select_topm_error_codes,
)
from hybrid_retrieval import hybrid_retrieve
from parsers import normalize_error_codes

MAX_RETRIEVE_ATTEMPTS = 2
RETRIEVE_TOP_K = 10
RERANK_TOP_N = 4
SUFFICIENCY_THRESHOLD = 0.35  # min top reranked score to call context "enough"


class ServingState(TypedDict, total=False):
    request_id: str
    question: str
    rewritten_question: str
    attempts: int
    detected_codes: List[str]
    code_lookup_attempted: bool
    exact_match_nodes: List[NodeWithScore]
    exact_nodes_by_code: Dict[str, List[NodeWithScore]]
    expanded_nodes: List[NodeWithScore]
    expansion_map: Dict[str, List[str]]
    missing_codes: List[str]
    retrieval_path: str  # "exact_match" | "exact_match+hybrid" | "hybrid"
    retrieved_nodes: List[NodeWithScore]
    reranked_nodes: List[NodeWithScore]
    sufficient: bool
    answer: str
    faithfulness_score: float
    citations: List[dict]


def rewrite_query(state: ServingState) -> dict:
    rid = state["request_id"]
    if state.get("attempts", 0) == 0:
        codes = normalize_error_codes(state["question"])
        if codes:
            print(f"[{rid}] detected error code(s) in question: {codes}", flush=True)
        print(f"[{rid}] attempt 1: using question as-is", flush=True)
        return {"rewritten_question": state["question"], "attempts": 1, "detected_codes": codes}

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
    rid = state["request_id"]
    if state.get("detected_codes") and not state.get("code_lookup_attempted"):
        print(f"[{rid}] routing to exact-match code lookup", flush=True)
        return "code_lookup"
    return "retrieve"


def code_lookup(state: ServingState) -> dict:
    rid = state["request_id"]
    codes = state["detected_codes"]
    found_nodes: List[NodeWithScore] = []
    missing_codes: List[str] = []
    exact_nodes_by_code: Dict[str, List[NodeWithScore]] = {}
    seen_ids = set()

    for code in codes:
        print(f"[{rid}] exact-match lookup for error_code={code}...", flush=True)
        nodes = fetch_nodes_by_error_code(code)
        print(f"[{rid}] exact-match lookup found {len(nodes)} chunk(s) for {code}", flush=True)
        if nodes:
            exact_nodes_by_code[code] = nodes
            for n in nodes:
                if n.node.node_id not in seen_ids:
                    found_nodes.append(n)
                    seen_ids.add(n.node.node_id)
        else:
            missing_codes.append(code)

    return {
        "reranked_nodes": found_nodes,
        "exact_match_nodes": found_nodes,
        "exact_nodes_by_code": exact_nodes_by_code,
        "missing_codes": missing_codes,
        "code_lookup_attempted": True,
        "retrieval_path": "exact_match",
    }


def route_after_code_lookup(state: ServingState) -> str:
    rid = state["request_id"]
    if state["reranked_nodes"] and not state["missing_codes"]:
        print(f"[{rid}] all detected code(s) matched -- skipping rerank/sufficiency, routing to expand_context", flush=True)
        return "expand_context"
    if state["missing_codes"]:
        print(
            f"[{rid}] code(s) {state['missing_codes']} code-shaped but matched no chunks in the "
            f"catalog -- falling back to hybrid retrieval to fill the gap "
            f"({len(state['reranked_nodes'])} chunk(s) already pinned from matched code(s))",
            flush=True,
        )
    return "retrieve"


def retrieve(state: ServingState) -> dict:
    rid = state["request_id"]
    print(f"[{rid}] hybrid retrieval (dense + sparse, fused) top-{RETRIEVE_TOP_K}...", flush=True)
    index = get_cached_index()
    nodes = hybrid_retrieve(
        state["rewritten_question"], index,
        dense_top_k=RETRIEVE_TOP_K, sparse_top_k=RETRIEVE_TOP_K,
    )
    print(f"[{rid}] retrieved {len(nodes)} chunk(s) after fusion", flush=True)
    return {"retrieved_nodes": nodes, "retrieval_path": "hybrid"}


def rerank(state: ServingState) -> dict:
    rid = state["request_id"]
    exact_nodes = state.get("exact_match_nodes") or []
    exact_ids = {n.node.node_id for n in exact_nodes}
    to_rerank = [n for n in state["retrieved_nodes"] if n.node.node_id not in exact_ids]
    remaining_budget = max(RERANK_TOP_N - len(exact_nodes), 0)

    reranked_rest: List[NodeWithScore] = []
    if to_rerank and remaining_budget:
        reranker = get_reranker(remaining_budget)
        reranked_rest = reranker.postprocess_nodes(to_rerank, query_str=state["rewritten_question"])

    reranked = exact_nodes + reranked_rest
    print(
        f"[{rid}] reranked to {len(reranked)} chunk(s) "
        f"({len(exact_nodes)} pinned exact-match + {len(reranked_rest)} from hybrid retrieval)",
        flush=True,
    )
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
        print(f"[{rid}] proceeding to expand_context", flush=True)
        return "expand_context"
    print(f"[{rid}] context insufficient, retrying retrieval", flush=True)
    return "rewrite_query"


def expand_context(state: ServingState) -> dict:
    rid = state["request_id"]
    exact_by_code = state.get("exact_nodes_by_code", {})
    matched_codes = [c for c in state.get("detected_codes", []) if exact_by_code.get(c)]

    if matched_codes:
        anchor_groups = {c: exact_by_code[c] for c in matched_codes}
        known_pools = {c: exact_by_code[c] for c in matched_codes}  # already complete + ordered
        trigger = "exact_match"
    else:
        top_codes = select_topm_error_codes(state["reranked_nodes"], WINDOWED_EXPANSION_TOP_M)
        if not top_codes:
            print(f"[{rid}] expand_context: no error-code-tagged chunks -- no-op", flush=True)
            return {"expanded_nodes": state["reranked_nodes"], "expansion_map": {}}
        anchor_groups = {}
        for n in state["reranked_nodes"]:
            code = n.node.metadata.get("error_code")
            if code in top_codes:
                anchor_groups.setdefault(code, []).append(n)
        known_pools = {}
        trigger = "topm_fallback"

    expanded, expansion_map = expand_sop_context(anchor_groups, known_pools, EXPANSION_CHAR_CAP)

    matched_ids = {n.node.node_id for group in anchor_groups.values() for n in group}
    passthrough = [
        n for n in state["reranked_nodes"]
        if n.node.node_id not in matched_ids and n.node.metadata.get("error_code") not in anchor_groups
    ]

    print(
        f"[{rid}] expand_context ({trigger}): {len(expansion_map)} code(s), "
        f"{sum(len(v) for v in expansion_map.values())} section(s) total", flush=True,
    )
    return {"expanded_nodes": passthrough + expanded, "expansion_map": expansion_map}


def generate(state: ServingState) -> dict:
    rid = state["request_id"]
    print(f"[{rid}] generating answer...", flush=True)
    llm = get_llm()
    context = build_numbered_context(state["expanded_nodes"])
    prompt = GENERATE_PROMPT.format(context=context, question=state["question"])
    answer = str(llm.chat([ChatMessage(role="user", content=prompt)])).strip()
    print(f"[{rid}] answer generated ({len(answer)} chars)", flush=True)
    citations = [
        {
            "node_id": n.node.node_id,
            "error_code": n.node.metadata.get("error_code"),
            "section": n.node.metadata.get("section"),
            "doc_type": n.node.metadata.get("doc_type"),
            "expanded": n.node.metadata.get("expanded", False),
            "score": n.score,
            "snippet": n.node.get_content()[:200],
        }
        for n in state["expanded_nodes"]
    ]
    return {"answer": answer, "citations": citations}


def faithfulness_check(state: ServingState) -> dict:
    from eval.metrics import faithfulness

    rid = state["request_id"]
    print(f"[{rid}] running faithfulness self-check...", flush=True)
    context_chunks = [n.node.get_content() for n in state["expanded_nodes"]]
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
    graph.add_node("expand_context", expand_context)
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
        {"expand_context": "expand_context", "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency",
        route_after_sufficiency,
        {"rewrite_query": "rewrite_query", "expand_context": "expand_context"},
    )
    graph.add_edge("expand_context", "generate")
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