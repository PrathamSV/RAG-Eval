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

import os

from dotenv import load_dotenv
from google.genai.types import EmbedContentConfig
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

EMBED_DIM = 768  # gemini-embedding-001 defaults to 3072; truncated via MRL
CHUNKS_TABLE = "rag_chunks"  # PGVectorStore stores this as table "data_rag_chunks"

EMBEDDING_MODEL = "gemini-embedding-001"

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
    """Point LlamaIndex's global Settings at Claude (generation) + Gemini
    (embedding). Safe to call more than once (idempotent)."""
    Settings.llm = get_llm(GENERATION_MODEL)
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name=EMBEDDING_MODEL,
        embedding_config=EmbedContentConfig(output_dimensionality=EMBED_DIM),
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