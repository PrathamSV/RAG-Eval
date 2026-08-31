"""
Retrieval + generation: embed the user's question, retrieve the top-k
matching chunks from pgvector, and generate an answer grounded in them.

This is the original phase-1 CLI script, kept as-is for quick manual testing.
For the corrective-RAG version with reranking, sufficiency checks, and
faithfulness scoring, see graphs/serving_graph.py (also reachable via
POST /query once the API is running).

Run: python query.py
"""

from llama_index.core import VectorStoreIndex

from common import configure_llamaindex_settings, get_vector_store

configure_llamaindex_settings()

vector_store = get_vector_store()


def main() -> None:
    index = VectorStoreIndex.from_vector_store(vector_store)

    query_engine = index.as_query_engine(similarity_top_k=5)

    question = input("Ask a question: ").strip()
    if not question:
        raise SystemExit("No question entered.")

    response = query_engine.query(question)

    print("\n--- Answer ---")
    print(response)

    print("\n--- Retrieved sources ---")
    for node in response.source_nodes:
        score = f"{node.score:.3f}" if node.score is not None else "n/a"
        snippet = node.node.get_content().replace("\n", " ")[:150]
        print(f"[score={score}] {snippet}...")


if __name__ == "__main__":
    main()
