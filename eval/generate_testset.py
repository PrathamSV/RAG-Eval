"""
Synthetic eval-set generation: for a sample of chunks already ingested into
pgvector, prompt the LLM to write a question that chunk alone can answer,
plus a short reference answer, and store (query, gold_chunk_id) pairs in
eval_queries.

Run: python -m eval.generate_testset --n 20
"""

import argparse
import json
import random
import re

import psycopg2
import psycopg2.extras
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

import db
from common import JUDGE_MODEL, get_pg_dsn

QUESTION_PROMPT = """Read the following passage and write ONE question that can be
answered using ONLY this passage. The question should be specific enough that
answering it requires this passage's information (not a vague question
answerable from general knowledge). Also give a short reference answer.

PASSAGE:
{chunk_text}

Respond with ONLY a JSON object: {{"question": "...", "reference_answer": "..."}}"""


def fetch_chunks(n: int) -> list:
    conn = psycopg2.connect(get_pg_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT node_id, text FROM data_rag_chunks")
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        raise SystemExit("No chunks found in data_rag_chunks — run ingest.py first.")
    return random.sample(rows, min(n, len(rows)))


def generate_question(chunk_text: str, llm: GoogleGenAI) -> dict | None:
    response = llm.chat(
        [ChatMessage(role="user", content=QUESTION_PROMPT.format(chunk_text=chunk_text))]
    )
    text = str(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main(n: int = 20) -> None:
    db.init_schema()
    llm = GoogleGenAI(model=JUDGE_MODEL)
    chunks = fetch_chunks(n)

    created = 0
    for i, row in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] requesting question for chunk {row['node_id']}...", flush=True)
        try:
            result = generate_question(row["text"], llm)
        except Exception as exc:
            print(f"[{i}/{len(chunks)}] FAILED: {exc!r}", flush=True)
            continue
        if not result or not result.get("question"):
            print(f"[{i}/{len(chunks)}] skipped (no parseable question)", flush=True)
            continue
        db.insert_eval_query(
            query_text=result["question"],
            gold_chunk_ids=[row["node_id"]],
            reference_answer=result.get("reference_answer"),
            query_type="single-hop",
        )
        created += 1
        print(f"[{i}/{len(chunks)}] ok — {created} created so far", flush=True)

    print(f"Generated {created} eval queries from {len(chunks)} sampled chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of chunks to sample")
    args = parser.parse_args()
    main(args.n)
