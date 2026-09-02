"""
Hybrid dense (vector) + sparse (Postgres full-text) retrieval with
Reciprocal Rank Fusion (RRF).

Used as the fallback path whenever the exact error-code lookup
(fetch_nodes_by_error_code) doesn't apply -- no code detected in the
question, or a well-formed code that isn't actually in the catalog. Plain
vector search alone under-retrieves on queries that reuse specific
terminology/phrasing from the source text but aren't semantically "close"
in embedding space -- the same failure mode noted in
fetch_nodes_by_error_code's docstring for opaque codes, just less extreme
for ordinary keyword-heavy phrasing. BM25-style full-text search catches
exact/near-exact term matches that dense retrieval misses; RRF fusion
combines the two ranked lists without needing to normalize scores across
two different scales (cosine distance vs ts_rank), which is the usual
pain point with hybrid search.

Requires ensure_fulltext_index() to have been run once against
data_rag_chunks (see common.py) -- called automatically at the end of
ingest.py and at API startup, so this is a no-op after the first setup.
"""

import psycopg2
import psycopg2.extras
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from common import get_pg_dsn

DENSE_TOP_K = 15
SPARSE_TOP_K = 15
RRF_K = 60  # standard smoothing constant -- de-emphasizes rank-1-vs-rank-2
            # noise while still rewarding a chunk that ranks highly in either list


def sparse_search(query: str, top_k: int = SPARSE_TOP_K) -> list[dict]:
    """Postgres full-text search (ts_rank over the generated `text_search`
    tsvector column) against data_rag_chunks. Returns rows with node_id,
    text, metadata_, score -- empty list if the query has no full-text
    matches (plainto_tsquery produces an empty tsquery for pure stopwords,
    or nothing in the corpus matches) rather than raising, since "no
    sparse hits" is an expected outcome fusion should handle gracefully,
    not an error."""
    conn = psycopg2.connect(get_pg_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT node_id, text, metadata_,
                       ts_rank(text_search, plainto_tsquery('english', %s)) AS score
                FROM data_rag_chunks
                WHERE text_search @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            return cur.fetchall()
    finally:
        conn.close()


def _sparse_rows_to_nodes(rows: list[dict]) -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(text=r["text"], id_=r["node_id"], metadata=r["metadata_"] or {}),
            score=float(r["score"]),
        )
        for r in rows
    ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]], k: int = RRF_K
) -> list[NodeWithScore]:
    """Merge N independently-ranked NodeWithScore lists into one, by rank
    position rather than raw score -- avoids normalizing dense
    cosine-distance scores against sparse ts_rank scores, which live on
    unrelated scales and would need corpus-specific calibration to compare
    directly. Each node's fused score is sum(1 / (k + rank)) across every
    list it appears in (rank is 1-indexed); a node absent from a list
    contributes 0 for that list. The final NodeWithScore.score is
    overwritten with the fused RRF score -- downstream code (rerank,
    sufficiency threshold) should treat it as a fusion rank, not a
    probability/similarity."""
    fused: dict[str, float] = {}
    node_lookup: dict[str, NodeWithScore] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            node_id = item.node.node_id
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (k + rank)
            # First occurrence wins for the underlying node/text -- dense
            # and sparse rows for the same node_id carry identical text
            # anyway (same source row in data_rag_chunks).
            if node_id not in node_lookup:
                node_lookup[node_id] = item

    merged = [
        NodeWithScore(node=node_lookup[node_id].node, score=score)
        for node_id, score in fused.items()
    ]
    merged.sort(key=lambda n: n.score, reverse=True)
    return merged


def hybrid_retrieve(
    query: str,
    index: VectorStoreIndex,
    dense_top_k: int = DENSE_TOP_K,
    sparse_top_k: int = SPARSE_TOP_K,
    rrf_k: int = RRF_K,
) -> list[NodeWithScore]:
    """Dense (vector) + sparse (full-text) retrieval, fused via RRF. `index`
    is passed in rather than built here so callers (serving_graph.py,
    eval_graph.py) reuse their own cached VectorStoreIndex instead of this
    module constructing/loading a second one."""
    dense_retriever = index.as_retriever(similarity_top_k=dense_top_k)
    dense_hits = dense_retriever.retrieve(query)

    sparse_rows = sparse_search(query, top_k=sparse_top_k)
    sparse_hits = _sparse_rows_to_nodes(sparse_rows)

    return reciprocal_rank_fusion([dense_hits, sparse_hits], k=rrf_k)