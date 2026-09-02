"""
Shared configuration: LLM, embedding model, and Postgres connection settings.

ingest.py, query.py, both LangGraph graphs, and the eval scripts all import
from here so the embedding model / dimension can never drift between
ingestion and querying (which would silently break retrieval).

Generation + judge run on Anthropic's Claude models. Anthropic does not
offer an embeddings API, so embeddings stay on Gemini
(`gemini-embedding-001`) — the two providers are independent and mixing
them is fine, since embeddings and generation never need to be the same
model/vendor.
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
                ORDER BY (metadata_->>'doc_type') ASC, id ASC
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

Passages:
{context}

Question: {question}

Answer:"""

def build_numbered_context(nodes) -> str:
    """Numbers retrieved/reranked nodes as [1], [2], ... for GENERATE_PROMPT,
    so the model can refer to and cross-reference specific passages instead
    of treating the context as one undifferentiated blob -- this is what
    makes the cross-passage combination rule in GENERATE_PROMPT usable."""
    return "\n\n".join(
        f"[{i}] {n.node.get_content()}" for i, n in enumerate(nodes, start=1)
    )
