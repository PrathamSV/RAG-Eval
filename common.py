"""
Shared configuration: LLM, embedding model, and Postgres connection settings.

ingest.py, query.py, both LangGraph graphs, and the eval scripts all import
from here so the embedding model / dimension can never drift between
ingestion and querying (which would silently break retrieval).

Generation + judge run on Anthropic's Claude models. Anthropic does not
offer an embeddings API, so embeddings run locally via HuggingFace
(`BAAI/bge-small-en-v1.5`, CPU-only — see EMBEDDING_MODEL below). This also
sidesteps the rate-limit fragility of a cloud embeddings API (e.g. Gemini's
free tier), at the cost of a smaller embedding dimension (384 vs 768).
"""

import json
import re
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

EMBED_DIM = 384  # bge-small-en-v1.5's native output dimension
CHUNKS_TABLE = "rag_chunks"  # PGVectorStore stores this as table "data_rag_chunks"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# --- Windowed expansion / per-SOP labeling ---------------------------------
# Total (not per-SOP) character budget for a single expand_context() call. Sized to
# comfortably fit the common 2-code multi-hop case (~15-20k chars per 5-page SOP) with
# headroom, while still bounding worst-case prompt growth when a question references
# many codes at once. This is a defensive backstop, not expected to bind in normal use.
EXPANSION_CHAR_CAP = 40_000

# Default number of top-ranked, error-code-tagged chunks to consider when no code was
# explicitly detected/matched in the question. See select_topm_error_codes() for exact
# semantics (raw chunk count, deduped by code afterward -- not "m distinct codes").
WINDOWED_EXPANSION_TOP_M = 1

# --- Generation / judge models -------------------------------------------
# Both are overridable via env var so you can swap models (e.g. to try a
# bigger Claude model, or point the judge at a different model than the
# generator) without touching any code. `.env` is the normal place to set
# these; the hardcoded defaults below are just fallbacks.
#
#   GENERATION_MODEL=claude-sonnet-5
#   JUDGE_MODEL=claude-opus-5
#
# See https://docs.claude.com/en/docs/about-claude/models for current model
# names/ids.
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "claude-haiku-4-5-20251001")

# Model used for judge tasks (faithfulness / relevance / correctness / synthetic
# question generation). Separated from the generation model so eval scoring can
# be swapped (e.g. to a stronger judge) without touching ingest/query.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GENERATION_MODEL)

# Max output tokens for the generation/judge LLM. Also overridable, since
# longer judge rationales or answers may need more room than the default.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


def get_llm(model: str = GENERATION_MODEL) -> Anthropic:
    """Build an Anthropic LLM client. Centralized here (rather than
    instantiated ad hoc) so every call site — generation, judge, eval,
    future graphs — picks up the same client configuration (max_tokens,
    api key resolution, etc.) automatically. Reads ANTHROPIC_API_KEY from
    the environment (set it in `.env`)."""
    return Anthropic(model=model, max_tokens=LLM_MAX_TOKENS)


def configure_llamaindex_settings() -> None:
    """Point LlamaIndex's global Settings at Claude (generation) + a local
    HuggingFace embedding model (embedding)."""
    Settings.llm = get_llm(GENERATION_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )


def get_vector_store() -> PGVectorStore:
    return PGVectorStore.from_params(
        database=os.getenv("PGDATABASE", "ragdb"),
        host=os.getenv("PGHOST", "localhost"),
        password=os.getenv("PGPASSWORD", "postgres"),
        port=os.getenv("PGPORT", "5432"),
        user=os.getenv("PGUSER", "postgres"),
        table_name=CHUNKS_TABLE,
        embed_dim=EMBED_DIM,
    )


_cached_index: VectorStoreIndex | None = None


def get_cached_index() -> VectorStoreIndex:
    """Shared VectorStoreIndex, built once per process. Previously
    graphs/serving_graph.py and graphs/eval_graph.py each kept a private
    `_cached_index` global -- harmless if only one graph ever runs, but
    api/main.py imports both into the same FastAPI process, so hitting
    /query and /eval/run each built (and kept in memory) their own copy of
    the HuggingFace embedding model. One cache here means one load,
    shared by both."""
    global _cached_index
    if _cached_index is None:
        configure_llamaindex_settings()
        _cached_index = VectorStoreIndex.from_vector_store(get_vector_store())
    return _cached_index


_reranker_cache: dict[int, object] = {}


def get_reranker(top_n: int):
    """Shared SentenceTransformerRerank cache, keyed by top_n -- same
    double-load issue as get_cached_index() above, this time for the
    cross-encoder weights. Import is local so common.py doesn't force a
    sentence-transformers dependency on callers that never rerank (e.g.
    generate_testset.py, hybrid_retrieval.py)."""
    from llama_index.core.postprocessor import SentenceTransformerRerank
    if top_n not in _reranker_cache:
        _reranker_cache[top_n] = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n
        )
    return _reranker_cache[top_n]


def extract_json(text: str) -> dict | None:
    """Pull the first {...} JSON object out of an LLM response and parse
    it. Shared by eval/generate_testset.py (question/reference-answer
    generation) and eval/metrics.py (judge scoring) -- both did the
    identical regex-then-json.loads independently. Returns None (rather
    than raising) on any parse failure, since a malformed judge/generator
    response should be skipped by the caller, not crash the run."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def get_pg_dsn() -> str:
    """Plain psycopg2 DSN for the eval tables. The chunks themselves live in
    the pgvector-managed table (data_rag_chunks, created by PGVectorStore);
    eval_queries/eval_runs/eval_results/feedback are plain tables we manage
    ourselves via psycopg2 in db.py."""
    return (
        f"host={os.getenv('PGHOST', 'localhost')} "
        f"port={os.getenv('PGPORT', '5432')} "
        f"dbname={os.getenv('PGDATABASE', 'ragdb')} "
        f"user={os.getenv('PGUSER', 'postgres')} "
        f"password={os.getenv('PGPASSWORD', 'postgres')}"
    )


def fetch_nodes_by_error_code(code: str) -> list[NodeWithScore]:
    """Exact-match lookup against the pgvector-managed chunks table by the
    `error_code` stored in each node's metadata (set at ingest time by
    parsers.py's parse_error_reference/parse_sop). Bypasses vector search
    entirely -- deterministic, instead of hoping the embedding kept the
    right chunk in the top-k. Opaque alphanumeric codes like "ABA0008"
    carry little embedding-relevant semantic signal, so two catalog rows
    for two *different* codes can end up closer together in vector space
    than the code's own reference row and SOP are to each other -- exactly
    the failure mode this bypasses.

    Returns every chunk tagged with this error_code -- the single catalog
    reference row (doc_type='reference') AND every SOP section
    (doc_type='sop') -- as NodeWithScore(score=1.0), so downstream code
    (generate(), citations, retrieval metrics) sees the same shape
    regardless of whether vector search or this exact-match path produced
    it. Returns an empty list if the code is well-formed but not actually
    in the ingested catalog (a typo'd digit, or a code from a differently
    versioned catalog) -- callers should treat that as "not a real code"
    and fall back to normal retrieval, not as an error.

    `metadata_` and `data_rag_chunks` are llama-index PGVectorStore's
    standard column/table names (see get_vector_store / CHUNKS_TABLE
    above) -- not something this function invents.
    """
    conn = psycopg2.connect(get_pg_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT node_id, text, metadata_
                FROM data_rag_chunks
                WHERE metadata_->>'error_code' = %s
                ORDER BY
                    (metadata_->>'doc_type') ASC,
                    COALESCE((metadata_->>'chunk_index')::int, id) ASC
                """,
                (code,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        NodeWithScore(
            node=TextNode(text=row["text"], id_=row["node_id"], metadata=row["metadata_"] or {}),
            score=1.0,
        )
        for row in rows
    ]


def select_topm_error_codes(ranked_nodes: list[NodeWithScore], m: int) -> list[str]:
    """Walk `ranked_nodes` in rank order, keep only chunks carrying a non-empty
    `error_code` metadata tag, take the top `m` of those BY CHUNK COUNT (not by
    distinct-code count), then dedupe by error_code preserving first-seen order.

    With the default m=1 this always yields at most one code. With m>1, if two of
    the top-m tagged chunks happen to share an error_code (e.g. two sections of the
    same SOP both rank highly), the result can have fewer than m distinct codes --
    this is intentional: a second hit on the same SOP is confirmatory evidence for
    that SOP, not license to also pull in an (m+1)-th, less relevant code.
    """
    tagged = [n for n in ranked_nodes if n.node.metadata.get("error_code")]
    seen: list[str] = []
    for n in tagged[:m]:
        code = n.node.metadata["error_code"]
        if code not in seen:
            seen.append(code)
    return seen


def _outward_fill_order(
    pool: list[NodeWithScore], seed_ids: set[str]
) -> list[NodeWithScore]:
    """`pool` entries not in `seed_ids`, ordered nearest-first by document position
    relative to the closest seed. `pool` is assumed already in canonical document
    order (as returned by fetch_nodes_by_error_code). This is what makes expansion
    grow outward from the anchor chunk(s) that actually triggered retrieval, instead
    of blindly filling from the top of the SOP."""
    seed_positions = [i for i, n in enumerate(pool) if n.node.node_id in seed_ids]
    if not seed_positions:
        seed_positions = [0]
    scored = [
        (min(abs(i - s) for s in seed_positions), i, n)
        for i, n in enumerate(pool)
        if n.node.node_id not in seed_ids
    ]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [n for _, _, n in scored]


def expand_sop_context(
    anchor_groups: dict[str, list[NodeWithScore]],
    known_pools: dict[str, list[NodeWithScore]],
    char_cap: int,
) -> tuple[list[NodeWithScore], dict[str, list[str]]]:
    """Expand each error code's anchor node(s) into its full SOP context, subject to
    a TOTAL character budget shared across every code in `anchor_groups` -- not a
    per-code budget.

    anchor_groups: error_code -> the node(s) that justified expanding this SOP
        (either the full exact-match set, or the 1+ chunk(s) a top-m scan surfaced).
    known_pools: error_code -> the FULL canonically-ordered node list for that code
        (reference row + every sop section), if the caller already has it. The
        exact-match path always has it (fetch_nodes_by_error_code already fetched
        everything). Omit a key to fetch it fresh; the top-m fallback path always
        omits, since it only has 1-2 anchor chunks, not the full SOP.
    char_cap: shared budget across ALL codes in anchor_groups combined.

    Every anchor node, plus each code's reference row (if present and not already an
    anchor), is always included regardless of budget -- expansion only ever trims how
    far it grows OUTWARD from that guaranteed core, never the core itself. If the
    combined mandatory content across all codes already exceeds char_cap, no further
    growth happens for any code and a warning is logged.

    Remaining budget is spent round-robin across codes -- one node from each code's
    outward-fill queue per round -- so no single code can exhaust the shared budget
    before others get a turn. A code's queue is permanently retired once its next
    candidate would exceed the remaining budget: growth is contiguous outward from
    the anchor, not a bin-packing search for whatever happens to fit.

    Returns (expanded_nodes, expansion_map). expansion_map is error_code -> ordered
    node_ids included for that code, used for save_details exports and logging --
    build_numbered_context does NOT need it; it re-derives grouping from each node's
    own error_code metadata.
    """
    pools: dict[str, list[NodeWithScore]] = {}
    included_ids: dict[str, set[str]] = {}
    remaining_budget = char_cap

    for code, anchors in anchor_groups.items():
        pool = known_pools.get(code) or fetch_nodes_by_error_code(code)
        pools[code] = pool

        anchor_ids = {n.node.node_id for n in anchors}
        reference_row = (
            pool[0] if pool and pool[0].node.metadata.get("doc_type") == "reference" else None
        )
        core = list(anchors)
        if reference_row and reference_row.node.node_id not in anchor_ids:
            core.append(reference_row)

        included_ids[code] = {n.node.node_id for n in core}
        remaining_budget -= sum(len(n.node.get_content()) for n in core)

    if remaining_budget < 0:
        print(
            f"WARNING: mandatory expansion content already exceeds EXPANSION_CHAR_CAP "
            f"({char_cap}) across {len(anchor_groups)} code(s) -- keeping all anchors/"
            f"reference rows, skipping further outward growth.",
            flush=True,
        )
        remaining_budget = 0

    queues = {c: _outward_fill_order(pools[c], included_ids[c]) for c in anchor_groups}
    pointers = {c: 0 for c in anchor_groups}
    active = [c for c in anchor_groups if queues[c]]

    while active and remaining_budget > 0:
        for code in list(active):
            i = pointers[code]
            if i >= len(queues[code]):
                active.remove(code)
                continue
            candidate = queues[code][i]
            cost = len(candidate.node.get_content())
            if cost > remaining_budget:
                active.remove(code)  # contiguous growth stops here for this code
                continue
            pointers[code] += 1
            included_ids[code].add(candidate.node.node_id)
            remaining_budget -= cost

    expansion_map: dict[str, list[str]] = {}
    expanded_nodes: list[NodeWithScore] = []
    for code, anchors in anchor_groups.items():
        anchor_id_set = {n.node.node_id for n in anchors}
        final = [n for n in pools[code] if n.node.node_id in included_ids[code]]
        for n in final:
            if n.node.node_id not in anchor_id_set:
                n.node.metadata["expanded"] = True
        expansion_map[code] = [n.node.node_id for n in final]
        expanded_nodes.extend(final)

    return expanded_nodes, expansion_map


def ensure_fulltext_index() -> bool:
    """Adds a generated tsvector column + GIN index to the pgvector-managed
    data_rag_chunks table, for hybrid_retrieval.py's sparse search. Safe to
    call repeatedly (checks to_regclass / column existence first). No-ops
    (returns False) if data_rag_chunks doesn't exist yet -- that just means
    nothing's been ingested, not an error.

    to_tsvector(regconfig, text) -- the two-argument form used below -- is
    IMMUTABLE, unlike the one-argument to_tsvector(text) which reads the
    default_text_search_config GUC and is only STABLE. Passing 'english'
    explicitly is what makes this legal inside a STORED generated column.
    """
    conn = psycopg2.connect(get_pg_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('data_rag_chunks')")
            if cur.fetchone()[0] is None:
                return False

            cur.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'data_rag_chunks' AND column_name = 'text_search'
                """
            )
            if cur.fetchone() is None:
                cur.execute(
                    """
                    ALTER TABLE data_rag_chunks
                    ADD COLUMN text_search tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_data_rag_chunks_text_search
                    ON data_rag_chunks USING GIN (text_search)
                    """
                )
                conn.commit()
        return True
    finally:
        conn.close()


GENERATE_PROMPT = """Answer the question using ONLY the passages below (numbered [1], [2], etc.). Some questions are answerable from a single passage; others require connecting specific facts stated in two or more different passages -- read all of them before deciding anything is missing.

Rules:
1. You may state a fact only if it is explicitly written in one of the passages, or is a direct, literal combination of facts each explicitly stated in the passages (e.g. passage [1] says X causes A, passage [2] says X causes B -- you may say X causes both A and B). Never add a reason, mechanism, or explanation that isn't itself written in the passages, even if it sounds plausible.
2. If the passages fully answer the question, answer it directly and completely.
3. If the passages answer part of the question but are missing one specific piece, answer the part that's supported and say exactly what's missing -- don't refuse the whole answer over one missing detail when the rest is solidly grounded.
4. If the passages don't contain enough to answer any part of the question, say so plainly. Do not fill the gap with outside knowledge or a plausible-sounding guess -- an honest "the passages don't cover this" always beats a confident but unsupported claim.
5. Before finalizing, check every claim you're about to make against the passages. If you can't point to where a specific claim comes from, cut it or flag it as unsupported rather than stating it as fact.
6. Some passages are grouped under a `=== SOP for error code X ===` banner and labeled with their owning error code and section. Passages under different banners belong to different, unrelated procedures -- never combine, average, or confuse step numbers or section names across two different error codes' SOPs, even when the section names look identical (e.g. two unrelated SOPs can each have their own "Step 2"). Always specify which error code's SOP a step belongs to when citing it.

Passages:
{context}

Question: {question}

Answer:"""

def _passage_label(n) -> str:
    """Every SOP-tagged passage is labeled with its error_code -- never a bare
    section name alone, since section names like 'Overview' or 'Resolution' recur
    verbatim across many different SOPs and are the single biggest ambiguity risk
    this feature is meant to prevent (the (code, section) pair is always unique;
    the section name alone is not)."""
    code = n.node.metadata.get("error_code")
    if not code:
        return ""
    if n.node.metadata.get("doc_type") == "reference":
        return f"(Catalog entry for {code})"
    section = n.node.metadata.get("section")
    return f"({code} — Section: {section!r})" if section else f"({code})"


def build_numbered_context(nodes) -> str:
    """Numbers nodes as [1], [2], ... for GENERATE_PROMPT, same as before. New:
    passages are grouped under a `=== SOP for error code X ===` banner whenever the
    error_code changes from the previous passage (nodes arrive pre-grouped by
    expand_context, so this is a single streaming pass, not a re-sort). Untagged
    passages (generic hybrid hits with no error_code) get no banner and no
    parenthetical label."""
    parts = []
    current_code = object()  # sentinel, guaranteed != any real code or None
    for i, n in enumerate(nodes, start=1):
        code = n.node.metadata.get("error_code")
        if code != current_code:
            if code:
                parts.append(f"=== SOP for error code {code} ===")
            current_code = code
        label = _passage_label(n)
        prefix = f"[{i}] {label}\n" if label else f"[{i}] "
        parts.append(f"{prefix}{n.node.get_content()}")
    return "\n\n".join(parts)
