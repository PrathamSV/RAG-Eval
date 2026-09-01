"""
Ingestion: load documents from ./data, chunk them, embed each chunk,
and write the vectors into a pgvector-backed table via LlamaIndex.

Run: python ingest.py
"""

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex

from common import configure_llamaindex_settings, get_vector_store

configure_llamaindex_settings()
Settings.chunk_size = 512 # 1024 default chunk size can be too large for multi-hop questions.
Settings.chunk_overlap = round(Settings.chunk_size * 0.15)

vector_store = get_vector_store()


def main() -> None:
    # Gemini's free tier is rate-limited (roughly 10 requests/min on
    # gemini-embedding-001 as of mid-2026) — fine for a handful of docs,
    # but add a delay between batches if you ingest a large corpus and
    # hit 429 errors.
    print("Loading documents from ./data...", flush=True)
    documents = SimpleDirectoryReader("./data").load_data()
    if not documents:
        raise SystemExit("No documents found in ./data — add a file and re-run.")
    print(f"Loaded {len(documents)} document(s). Connecting to pgvector...", flush=True)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(
        f"Chunking (size={Settings.chunk_size}, overlap={Settings.chunk_overlap}) "
        "and embedding into pgvector — progress below:",
        flush=True,
    )
    # from_documents handles chunking (via the default node parser),
    # embedding each chunk, and writing rows into the pgvector table.
    # show_progress=True renders a per-node tqdm bar so a large corpus
    # doesn't look stuck while it embeds.
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Done. Ingested {len(documents)} document(s) into pgvector table 'rag_chunks'.", flush=True)


if __name__ == "__main__":
    main()