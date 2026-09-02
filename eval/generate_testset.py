"""
Synthetic eval-set generation: builds eval_queries rows from the chunks
already ingested into pgvector.

Generates two kinds of eval queries, mixed together:

  - single-hop: one question per sampled chunk, answerable from that chunk
    alone. gold_chunk_ids has exactly 1 entry.
  - multi-hop (semantic): a question spanning two chunks that are
    thematically related (by embedding similarity) but NOT adjacent in the
    document — a stronger test of retrieval's ability to find scattered
    evidence, since the two chunks won't naturally sit next to each other
    in a top-k retrieval the way adjacent chunks tend to. gold_chunk_ids
    has 2 entries.

Every multi-hop candidate is verified with a follow-up judge call before
being kept: ...

Run:
    python -m eval.generate_testset --n 20
    python -m eval.generate_testset --n 20 --multi-hop-fraction 0.3
"""

import argparse
import random

import psycopg2
import psycopg2.extras
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic

import db
from common import JUDGE_MODEL, extract_json, get_llm, get_pg_dsn

SINGLE_HOP_PROMPT = """Read the following passage and write ONE question that can be
answered using ONLY this passage. The question should be specific enough that
answering it requires this passage's information (not a vague question
answerable from general knowledge). Also give a short reference answer.

PASSAGE:
{chunk_text}

Respond with ONLY a JSON object: {{"question": "...", "reference_answer": "..."}}"""

MULTI_HOP_PROMPT = """Read the following two passages from the same document. They likely
share some overlapping context, but each ALSO contains at least one specific
detail, claim, fact, or number that the OTHER passage does NOT contain.

Follow these steps in order:

1. Find ONE specific, concrete detail in Passage A that is NOT stated or
   implied anywhere in Passage B. Prefer something concrete over something
   general — a number, a name, a specific mechanism, a specific outcome —
   over a vague theme or topic.
2. Find ONE specific, concrete detail in Passage B that is NOT stated or
   implied anywhere in Passage A, independent of the detail you picked
   from Passage A.
3. Write ONE question that CANNOT be answered without BOTH of those exact
   details. The question should force the reader to connect them — e.g.
   "How does [detail from A] affect [detail from B]?", "What is the
   relationship between [detail from A] and [detail from B]?", or "Given
   [detail from A], what does [detail from B] imply?" — rather than a
   general question that both passages happen to be relevant to.

A question is INVALID if:
- Either passage ALONE already contains everything needed to answer it,
  even if the other passage discusses a related topic.
- The two details you picked are really the same fact restated, or one
  implies the other.
- The question could be answered in general terms without citing the
  specific detail from each passage (e.g. "how do these two things
  relate" without needing the numbers/specifics themselves).

If you can't find two such details, pick a narrower, more specific pair
rather than falling back to a general question about the shared topic.

PASSAGE A:
{chunk_a}

PASSAGE B:
{chunk_b}

Respond with ONLY a JSON object:
{{"detail_a": "<the specific detail from Passage A, in your own words>",
  "detail_b": "<the specific detail from Passage B, in your own words>",
  "question": "...",
  "reference_answer": "..."}}"""

MULTI_HOP_RETRY_PROMPT = """You previously wrote this question over the two passages below,
intending it to require BOTH passages:

PREVIOUS QUESTION: {previous_question}

It was rejected: {reason}

This usually means the details you picked weren't actually independent —
one passage alone was enough to answer it. Try again, more strictly:

1. Find a DIFFERENT specific, concrete detail in Passage A — a number,
   name, mechanism, or outcome — that has NO trace, restatement, or
   implication in Passage B at all.
2. Find a DIFFERENT specific, concrete detail in Passage B with no trace,
   restatement, or implication in Passage A.
3. Double check: could someone answer your planned question from Passage A
   alone, using only general knowledge or inference, without ever reading
   Passage B? If yes, pick a different detail from A. Same check for B.
4. Write ONE question that requires connecting exactly those two details.

PASSAGE A:
{chunk_a}

PASSAGE B:
{chunk_b}

Respond with ONLY a JSON object:
{{"detail_a": "<the specific detail from Passage A, in your own words>",
  "detail_b": "<the specific detail from Passage B, in your own words>",
  "question": "...",
  "reference_answer": "..."}}"""

VERIFY_PROMPT = """A question was supposedly written so that it requires BOTH of the
passages below to answer fully. Check whether that's actually true — be
strict, since a question answerable from just one passage should NOT count
as multi-hop.

QUESTION:
{question}

PASSAGE A (alone):
{chunk_a}

PASSAGE B (alone):
{chunk_b}

Respond with ONLY a JSON object:
{{"answerable_from_a_alone": true/false, "answerable_from_b_alone": true/false, "reason": "<one sentence>"}}"""


# Skip the closest matches (near-duplicates in meaning -- exactly what
# tends to fail multi-hop verification, since one chunk alone can already
# answer a question about a near-identical chunk) and sample the pairing
# partner from a window a bit further out instead.
SEMANTIC_PAIR_RANK_SKIP = 2    # 0-indexed offset: skip ranks 1-2 (closest matches)
SEMANTIC_PAIR_RANK_WINDOW = 4  # then sample from the next 6 (ranks 3-8)


def fetch_all_chunks() -> list:
    """All ingested chunks, in document order. Insertion order (the `id`
    serial column) tracks document order for a single-document corpus,
    since the node parser chunks top-to-bottom in one pass — that's what
    makes 'adjacent chunk' a meaningful notion here without needing to add
    ordering metadata at ingest time."""
    conn = psycopg2.connect(get_pg_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, node_id, text FROM data_rag_chunks ORDER BY id")
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        raise SystemExit("No chunks found in data_rag_chunks — run ingest.py first.")
    return rows


def fetch_semantic_pair(anchor_id: int, exclude_ids: set) -> dict | None:
    """A chunk semantically related to `anchor_id`, but NOT its single
    closest match. The nearest neighbor by cosine distance tends to be
    near-duplicate in meaning -- same topic, often the same facts restated
    -- which is exactly what fails multi-hop verification (one passage
    alone ends up sufficient to answer the question). This instead pulls a
    window of the next-closest candidates (ranks 3-8 by default, on top of
    excluding the anchor + its immediate neighbors via `exclude_ids`) and
    picks one at random -- related enough that a multi-hop question over
    the pair still makes sense, but with more room for each chunk to carry
    a detail the other doesn't restate.

    Falls back to the single closest match if the corpus is too small to
    have any candidates past the skipped ranks."""
    conn = psycopg2.connect(get_pg_dsn())
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT node_id, text
                FROM data_rag_chunks
                WHERE id != %s AND NOT (id = ANY(%s))
                ORDER BY embedding <=> (SELECT embedding FROM data_rag_chunks WHERE id = %s)
                OFFSET %s LIMIT %s
                """,
                (
                    anchor_id,
                    list(exclude_ids),
                    anchor_id,
                    SEMANTIC_PAIR_RANK_SKIP,
                    SEMANTIC_PAIR_RANK_WINDOW,
                ),
            )
            candidates = cur.fetchall()

            if not candidates:
                # Small corpus -- nothing left past the skipped ranks.
                # Fall back to the single closest match rather than
                # returning None and losing this anchor entirely.
                cur.execute(
                    """
                    SELECT node_id, text
                    FROM data_rag_chunks
                    WHERE id != %s AND NOT (id = ANY(%s))
                    ORDER BY embedding <=> (SELECT embedding FROM data_rag_chunks WHERE id = %s)
                    LIMIT 1
                    """,
                    (anchor_id, list(exclude_ids), anchor_id),
                )
                candidates = cur.fetchall()

            return random.choice(candidates) if candidates else None
    finally:
        conn.close()


def _ask_json(llm: Anthropic, prompt: str) -> dict | None:
    """Send `prompt`, parse the first {...} JSON object out of the reply."""
    response = llm.chat([ChatMessage(role="user", content=prompt)])
    return extract_json(str(response))


def generate_single_hop(chunk_text: str, llm: Anthropic) -> dict | None:
    return _ask_json(llm, SINGLE_HOP_PROMPT.format(chunk_text=chunk_text))


def generate_multi_hop_candidate(chunk_a_text: str, chunk_b_text: str, llm: Anthropic) -> dict | None:
    return _ask_json(llm, MULTI_HOP_PROMPT.format(chunk_a=chunk_a_text, chunk_b=chunk_b_text))


def regenerate_multi_hop_candidate(
    chunk_a_text: str, chunk_b_text: str, previous_question: str, reason: str, llm: Anthropic
) -> dict | None:
    prompt = MULTI_HOP_RETRY_PROMPT.format(
        previous_question=previous_question, reason=reason, chunk_a=chunk_a_text, chunk_b=chunk_b_text
    )
    return _ask_json(llm, prompt)


def verify_multi_hop(question: str, chunk_a_text: str, chunk_b_text: str, llm: Anthropic) -> dict | None:
    prompt = VERIFY_PROMPT.format(question=question, chunk_a=chunk_a_text, chunk_b=chunk_b_text)
    return _ask_json(llm, prompt)


def _generate_and_store_multi_hop(
    chunk_a: dict, chunk_b: dict, llm: Anthropic, i: int, total: int, max_attempts: int = 2
) -> int:
    """... (docstring unchanged) ..."""
    previous_question = None
    previous_reason = None


    for attempt in range(1, max_attempts + 1):
        try:
            if attempt == 1:
                candidate = generate_multi_hop_candidate(chunk_a["text"], chunk_b["text"], llm)
            else:
                candidate = regenerate_multi_hop_candidate(
                    chunk_a["text"], chunk_b["text"], previous_question, previous_reason, llm
                )
        except Exception as exc:
            print(f"[{i}/{total}] FAILED generating (attempt {attempt}): {exc!r}", flush=True)
            return 0
        if not candidate or not candidate.get("question"):
            print(f"[{i}/{total}] skipped (no parseable question, attempt {attempt})", flush=True)
            return 0
        print(f"[{i}/{total}] candidate (attempt {attempt}): {candidate['question']!r}", flush=True)

        print(f"[{i}/{total}] verifying question genuinely needs both passages...", flush=True)
        try:
            verdict = verify_multi_hop(candidate["question"], chunk_a["text"], chunk_b["text"], llm)
        except Exception as exc:
            print(f"[{i}/{total}] verification FAILED: {exc!r} — discarding to be safe", flush=True)
            return 0
        if verdict is None:
            print(f"[{i}/{total}] verification unparseable — discarding to be safe", flush=True)
            return 0

        # Fail safe: if either flag is missing/ambiguous, treat it as "yes,
        # answerable alone" so the candidate gets discarded rather than kept.
        answerable_alone = verdict.get("answerable_from_a_alone", True) or verdict.get(
            "answerable_from_b_alone", True
        )
        reason = verdict.get("reason", "(none given)")
        print(
            f"[{i}/{total}] verdict: a_alone={verdict.get('answerable_from_a_alone')!r} "
            f"b_alone={verdict.get('answerable_from_b_alone')!r} reason={reason!r}",
            flush=True,
        )

        if not answerable_alone:
            db.insert_eval_query(
                query_text=candidate["question"],
                gold_chunk_ids=[chunk_a["node_id"], chunk_b["node_id"]],
                reference_answer=candidate.get("reference_answer"),
                query_type="multi-hop",
            )
            print(
                f"[{i}/{total}] ok — multi-hop (semantic) query created (attempt {attempt})", flush=True
            )
            return 1

        if attempt < max_attempts:
            print(f"[{i}/{total}] rejected (attempt {attempt}) — retrying with feedback", flush=True)
            previous_question = candidate["question"]
            previous_reason = reason
        else:
            print(f"[{i}/{total}] rejected (attempt {attempt}) — giving up on this pair", flush=True)

    return 0


def main(n: int = 20, multi_hop_fraction: float = 0.5) -> None:
    """
    n: total number of eval queries to attempt to create.
    multi_hop_fraction: fraction of n that should be multi-hop questions
        (the rest are single-hop). Default 0.5 (~50/50 split). All
        multi-hop questions are generated over semantically related but
        non-adjacent chunk pairs (found via embedding similarity over
        data_rag_chunks) — this is a stronger test of retrieval than
        adjacent-chunk pairing, since the two gold chunks won't naturally
        sit next to each other in a top-k retrieval the way adjacent
        chunks tend to.
    """
    db.init_schema()
    llm = get_llm(model=JUDGE_MODEL)
    chunks = fetch_all_chunks()
    if len(chunks) < 3 and multi_hop_fraction > 0:
        print(
            "Fewer than 3 chunks ingested — can't find non-adjacent semantic "
            "pairs for multi-hop questions. Falling back to single-hop only.",
            flush=True,
        )
        multi_hop_fraction = 0.0

    n_multi = round(n * multi_hop_fraction)
    n_single = n - n_multi
    total = n_single + n_multi

    print(
        f"Target: {n_single} single-hop, {n_multi} multi-hop (semantic) — "
        f"{total} attempt(s) total.",
        flush=True,
    )

    created = 0
    attempted = 0

    # --- single-hop ---------------------------------------------------
    for row in random.sample(chunks, min(n_single, len(chunks))):
        attempted += 1
        print(f"[{attempted}/{total}] single-hop: chunk {row['node_id']}...", flush=True)
        try:
            result = generate_single_hop(row["text"], llm)
        except Exception as exc:
            print(f"[{attempted}/{total}] FAILED: {exc!r}", flush=True)
            continue
        if not result or not result.get("question"):
            print(f"[{attempted}/{total}] skipped (no parseable question)", flush=True)
            continue
        db.insert_eval_query(
            query_text=result["question"],
            gold_chunk_ids=[row["node_id"]],
            reference_answer=result.get("reference_answer"),
            query_type="single-hop",
        )
        created += 1
        print(f"[{attempted}/{total}] ok — {created} created so far", flush=True)

    # --- multi-hop: semantic (non-adjacent) -----------------------------
    if n_multi > 0 and len(chunks) >= 3:
        anchor_idxs = random.sample(range(len(chunks)), min(n_multi, len(chunks)))
        for idx in anchor_idxs:
            attempted += 1
            anchor = chunks[idx]
            exclude_ids = {anchor["id"]}
            if idx > 0:
                exclude_ids.add(chunks[idx - 1]["id"])
            if idx < len(chunks) - 1:
                exclude_ids.add(chunks[idx + 1]["id"])

            partner = fetch_semantic_pair(anchor["id"], exclude_ids)
            if partner is None:
                print(
                    f"[{attempted}/{total}] multi-hop (semantic): "
                    f"no non-adjacent partner found for {anchor['node_id']}, skipping",
                    flush=True,
                )
                continue
            print(
                f"[{attempted}/{total}] multi-hop (semantic): "
                f"chunks {anchor['node_id']} + {partner['node_id']}...",
                flush=True,
            )
            created += _generate_and_store_multi_hop(anchor, partner, llm, attempted, total)
    elif n_multi > 0:
        print("Not enough chunks for semantic multi-hop pairs — skipping.", flush=True)

    print(
        f"Generated {created} eval quer{'y' if created == 1 else 'ies'} "
        f"from {attempted} attempt(s) "
        f"({n_single} single-hop / {n_multi} semantic multi-hop targeted)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Total number of eval queries to generate")
    parser.add_argument(
        "--multi-hop-fraction",
        type=float,
        default=0.5,
        help="Fraction of --n that should be multi-hop questions (default 0.5)",
    )
    args = parser.parse_args()
    main(args.n, args.multi_hop_fraction)