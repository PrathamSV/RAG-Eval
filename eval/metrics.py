"""
Retrieval metrics (Hit Rate@K, Recall@K, Precision@K, MRR) computed directly
from ranked chunk ids vs. a gold set, plus LLM-judge metrics (faithfulness,
answer relevance, answer correctness) that score generation quality.
"""

import json
import re
from typing import Sequence

from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI

from common import JUDGE_MODEL

_judge_llm = None


def _get_judge_llm() -> GoogleGenAI:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = GoogleGenAI(model=JUDGE_MODEL)
    return _judge_llm


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def hit_rate_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """1.0 if ANY gold chunk shows up anywhere in the retrieved set, else 0.0."""
    gold = set(gold_ids)
    return 1.0 if any(rid in gold for rid in retrieved_ids) else 0.0


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Fraction of gold chunks that were retrieved."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    hit = len(gold.intersection(retrieved_ids))
    return hit / len(gold)


def precision_at_k(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Fraction of retrieved chunks that were gold."""
    if not retrieved_ids:
        return 0.0
    gold = set(gold_ids)
    hit = sum(1 for rid in retrieved_ids if rid in gold)
    return hit / len(retrieved_ids)


def mrr(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    """Reciprocal rank of the first gold chunk found, 0.0 if none found."""
    gold = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def retrieval_metrics(retrieved_ids: Sequence[str], gold_ids: Sequence[str]) -> dict:
    return {
        "hit_rate": hit_rate_at_k(retrieved_ids, gold_ids),
        "recall": recall_at_k(retrieved_ids, gold_ids),
        "precision": precision_at_k(retrieved_ids, gold_ids),
        "mrr": mrr(retrieved_ids, gold_ids),
    }


# ---------------------------------------------------------------------------
# LLM-judge generation metrics
# ---------------------------------------------------------------------------


def _judge_score(prompt: str) -> float:
    """Ask the judge LLM for a JSON object {"score": 0-1, "reason": "..."}
    and return the score. Defaults to 0.0 on any parse failure so a flaky
    judge response degrades a metric rather than crashing an eval run."""
    llm = _get_judge_llm()
    response = llm.chat([ChatMessage(role="user", content=prompt)])
    text = str(response)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return 0.0
    try:
        data = json.loads(match.group(0))
        return float(data.get("score", 0.0))
    except (ValueError, json.JSONDecodeError, TypeError):
        return 0.0


def faithfulness(answer: str, context_chunks: Sequence[str]) -> float:
    """Does the answer only make claims supported by the retrieved context?
    (a.k.a. hallucination check)."""
    context = "\n---\n".join(context_chunks)
    prompt = f"""You are grading whether an ANSWER is faithful to the given CONTEXT
(i.e. every claim in the answer is supported by the context, with no
hallucinated facts).

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
where 1.0 means fully faithful and 0.0 means entirely unsupported."""
    return _judge_score(prompt)


def answer_relevance(question: str, answer: str) -> float:
    """Does the answer actually address the question asked (independent of
    whether it's factually correct)?"""
    prompt = f"""You are grading whether an ANSWER is relevant to the QUESTION asked.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
where 1.0 means directly and fully relevant, 0.0 means off-topic."""
    return _judge_score(prompt)


def answer_correctness(answer: str, reference_answer: str) -> float:
    """Does the answer match the reference answer's factual content?"""
    if not reference_answer:
        return 0.0
    prompt = f"""You are grading whether a CANDIDATE answer is factually
correct compared to a REFERENCE answer.

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
where 1.0 means factually equivalent, 0.0 means contradicts or misses the key fact."""
    return _judge_score(prompt)


def generation_metrics(
    question: str, answer: str, context_chunks: Sequence[str], reference_answer: str | None = None
) -> dict:
    return {
        "faithfulness": faithfulness(answer, context_chunks),
        "answer_relevance": answer_relevance(question, answer),
        "answer_correctness": (
            answer_correctness(answer, reference_answer) if reference_answer else None
        ),
    }
