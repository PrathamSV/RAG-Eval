"""
Thin psycopg2 helpers for the eval_queries / eval_runs / eval_results /
feedback tables (schema.sql). Kept separate from LlamaIndex's PGVectorStore,
which already owns the chunks table.
"""

import json
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from common import get_pg_dsn

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


@contextmanager
def get_conn():
    conn = psycopg2.connect(get_pg_dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema() -> None:
    """Create eval_queries/eval_runs/eval_results/feedback if they don't exist yet."""
    with open(_SCHEMA_PATH) as f:
        ddl = f.read()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)


def insert_eval_query(
    query_text: str,
    gold_chunk_ids: list,
    reference_answer: str | None = None,
    query_type: str = "single-hop",
) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_queries (query_text, gold_chunk_ids, reference_answer, query_type)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (query_text, gold_chunk_ids, reference_answer, query_type),
            )
            return cur.fetchone()[0]


def fetch_eval_queries() -> list:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM eval_queries ORDER BY id")
            return cur.fetchall()


def create_eval_run(config: dict) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO eval_runs (config) VALUES (%s) RETURNING id",
                (json.dumps(config),),
            )
            return cur.fetchone()[0]


def finish_eval_run(run_id: int, status: str = "complete") -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE eval_runs SET finished_at = now(), status = %s WHERE id = %s",
                (status, run_id),
            )


def insert_eval_result(
    run_id: int,
    query_id: int,
    retrieved_chunk_ids: list,
    latency_ms: float,
    metrics: dict,
    generated_answer: str,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_results
                   (run_id, query_id, retrieved_chunk_ids, latency_ms, hit_rate, recall,
                    precision, mrr, generated_answer, faithfulness, answer_relevance, answer_correctness)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    run_id,
                    query_id,
                    retrieved_chunk_ids,
                    latency_ms,
                    metrics.get("hit_rate"),
                    metrics.get("recall"),
                    metrics.get("precision"),
                    metrics.get("mrr"),
                    generated_answer,
                    metrics.get("faithfulness"),
                    metrics.get("answer_relevance"),
                    metrics.get("answer_correctness"),
                ),
            )


def fetch_run_summary(run_id: int):
    """Returns (run_row, aggregated_metrics_row) or (None, None) if not found."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM eval_runs WHERE id = %s", (run_id,))
            run = cur.fetchone()
            if run is None:
                return None, None
            cur.execute(
                """SELECT
                       avg(hit_rate)           AS avg_hit_rate,
                       avg(recall)             AS avg_recall,
                       avg(precision)          AS avg_precision,
                       avg(mrr)                AS avg_mrr,
                       avg(faithfulness)       AS avg_faithfulness,
                       avg(answer_relevance)   AS avg_answer_relevance,
                       avg(answer_correctness) AS avg_answer_correctness,
                       avg(latency_ms)         AS avg_latency_ms,
                       count(*)                AS n_queries
                   FROM eval_results WHERE run_id = %s""",
                (run_id,),
            )
            agg = cur.fetchone()
            return run, agg


def fetch_run_details(run_id: int) -> list:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT r.*, q.query_text, q.gold_chunk_ids, q.query_type
                   FROM eval_results r JOIN eval_queries q ON q.id = r.query_id
                   WHERE r.run_id = %s ORDER BY r.id""",
                (run_id,),
            )
            return cur.fetchall()


def insert_feedback(
    query_text: str, answer_text: str, rating: int, retrieved_chunk_ids: list | None = None
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO feedback (query_text, answer_text, rating, retrieved_chunk_ids)
                   VALUES (%s, %s, %s, %s)""",
                (query_text, answer_text, rating, retrieved_chunk_ids),
            )
