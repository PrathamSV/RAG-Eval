"""
Ingestion: load documents from ./data, chunk them, embed each chunk,
and write the vectors into a pgvector-backed table via LlamaIndex.

Run: python ingest.py
"""

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex

from common import configure_llamaindex_settings, get_vector_store

configure_llamaindex_settings()

# Default chunk_size is 1024 tokens — bigger than our whole sample doc, so
# nothing would split and every query would retrieve the same single chunk.
# Shrink it so retrieval actually has multiple candidates to choose from.
Settings.chunk_size = 200
Settings.chunk_overlap = 20

vector_store = get_vector_store()


def main() -> None:
    # Gemini's free tier is rate-limited (roughly 10 requests/min on
    # gemini-embedding-001 as of mid-2026) — fine for a handful of docs,
    # but add a delay between batches if you ingest a large corpus and
    # hit 429 errors.
    documents = SimpleDirectoryReader("./data").load_data()
    if not documents:
        raise SystemExit("No documents found in ./data — add a file and re-run.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # from_documents handles chunking (via the default node parser),
    # embedding each chunk, and writing rows into the pgvector table.
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"Ingested {len(documents)} document(s) into pgvector table 'rag_chunks'.")


if __name__ == "__main__":
    main()
