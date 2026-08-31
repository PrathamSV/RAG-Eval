"""
Shared configuration: LLM, embedding model, and Postgres connection settings.

ingest.py, query.py, both LangGraph graphs, and the eval scripts all import
from here so the embedding model / dimension can never drift between
ingestion and querying (which would silently break retrieval).
"""

import os

from dotenv import load_dotenv
from google.genai.types import EmbedContentConfig
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

EMBED_DIM = 768  # gemini-embedding-001 defaults to 3072; truncated via MRL
CHUNKS_TABLE = "rag_chunks"  # PGVectorStore stores this as table "data_rag_chunks"

GENERATION_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Model used for judge tasks (faithfulness / relevance / correctness / synthetic
# question generation). Separated from the generation model so eval scoring can
# be swapped (e.g. to a stronger judge) without touching ingest/query.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", GENERATION_MODEL)


def configure_llamaindex_settings() -> None:
    """Point LlamaIndex's global Settings at Gemini flash + the embedding model.
    Safe to call more than once (idempotent)."""
    Settings.llm = GoogleGenAI(model=GENERATION_MODEL)
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
