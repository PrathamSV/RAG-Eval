"""
Retrieval metrics (Hit Rate@K, Recall@K, Precision@K, MRR) computed directly
from ranked chunk ids vs. a gold set, plus LLM-judge metrics (faithfulness,
answer relevance, answer correctness) that score generation quality.
"""

from typing import Sequence
import numpy as np
from scipy import stats as scipy_stats

from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic

from common import JUDGE_MODEL, extract_json, get_llm

_judge_llm = None


def _get_judge_llm() -> Anthropic:
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = get_llm(model=JUDGE_MODEL)
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


def _judge_score(prompt: str, label: str) -> dict:
    """Send `prompt` to the judge LLM and parse a {"score":.., "reason":..}
    JSON object out of the reply.

    Returns {"score": float, "reason": str | None, "parse_failed": bool}.
    `parse_failed=True` covers both failure modes below (no parseable JSON at
    all, or a non-numeric "score" field) -- in either case `score` falls back
    to 0.0, but that 0.0 is a symptom of a broken judge call, not necessarily
    a genuine "entirely unsupported" / "fully incorrect" verdict. Before this
    field existed, the two were indistinguishable in stored results: a
    faithfulness=0.0 in eval_results/save_details could mean either "the
    judge said so" or "the judge's response didn't parse." Callers that only
    need the number (faithfulness/answer_relevance/answer_correctness below)
    discard reason/parse_failed; generation_metrics keeps them specifically
    so save_details can carry the distinction (see EvalQueryState's
    judge_details / eval_graph.py's compute_generation_metrics)."""
    print(f"    judging {label}...", flush=True)
    llm = _get_judge_llm()
    response = llm.chat([ChatMessage(role="user", content=prompt)])
    data = extract_json(str(response))
    if data is None:
        print(f"    judging {label}: no parseable JSON in judge response — scoring 0.0", flush=True)
        return {"score": 0.0, "reason": None, "parse_failed": True}
    try:
        score = float(data.get("score", 0.0))
    except (ValueError, TypeError):
        print(f"    judging {label}: malformed JSON in judge response — scoring 0.0", flush=True)
        return {"score": 0.0, "reason": data.get("reason"), "parse_failed": True}
    reason = data.get("reason")
    print(f"    judging {label}: {score} ({reason!r})", flush=True)
    return {"score": score, "reason": reason, "parse_failed": False}


def _faithfulness_prompt(answer: str, context_chunks: Sequence[str]) -> str:
    context = "\n---\n".join(context_chunks)
    return f"""You are grading whether an ANSWER is faithful to the given CONTEXT
(i.e. every claim in the answer is supported by the context, with no
hallucinated facts).

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
where 1.0 means fully faithful and 0.0 means entirely unsupported."""


def _answer_relevance_prompt(question: str, answer: str) -> str:
    return f"""You are grading whether an ANSWER is relevant to the QUESTION asked.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
where 1.0 means directly and fully relevant, 0.0 means off-topic."""


def _answer_correctness_prompt(answer: str, reference_answer: str) -> str:
    return f"""You are grading whether a CANDIDATE answer is factually
correct compared to a REFERENCE answer.

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{answer}

Respond with ONLY a JSON object: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
where 1.0 means factually equivalent, 0.0 means contradicts or misses the key fact."""


def faithfulness(answer: str, context_chunks: Sequence[str]) -> float:
    """Does the answer only make claims supported by the retrieved context?
    (a.k.a. hallucination check)."""
    return _judge_score(_faithfulness_prompt(answer, context_chunks), label="faithfulness")["score"]


def answer_relevance(question: str, answer: str) -> float:
    """Does the answer actually address the question asked (independent of
    whether it's factually correct)?"""
    return _judge_score(_answer_relevance_prompt(question, answer), label="answer_relevance")["score"]


def answer_correctness(answer: str, reference_answer: str) -> float:
    """Does the answer match the reference answer's factual content?"""
    if not reference_answer:
        return 0.0
    return _judge_score(_answer_correctness_prompt(answer, reference_answer), label="answer_correctness")["score"]


def generation_metrics(
    question: str, answer: str, context_chunks: Sequence[str], reference_answer: str | None = None
) -> dict:
    """As before, plus a "judge_details" key: per-metric {reason,
    parse_failed}, so a 0.0 score can be triaged as a genuine judge verdict
    vs. a silently-defaulted parse failure (see _judge_score's docstring).
    Each metric still costs exactly one LLM call -- judge_details is
    populated from the same _judge_score call that produces the score, not
    an extra round-trip. compute_generation_metrics (eval_graph.py) pops
    "judge_details" back out before it reaches eval_results/db, keeping the
    numeric metrics dict unchanged for every existing reader; it only flows
    into the save_details JSON export."""
    faithfulness_result = _judge_score(_faithfulness_prompt(answer, context_chunks), label="faithfulness")
    relevance_result = _judge_score(_answer_relevance_prompt(question, answer), label="answer_relevance")
    correctness_result = (
        _judge_score(_answer_correctness_prompt(answer, reference_answer), label="answer_correctness")
        if reference_answer
        else None
    )

    return {
        "faithfulness": faithfulness_result["score"],
        "answer_relevance": relevance_result["score"],
        "answer_correctness": correctness_result["score"] if correctness_result else None,
        "judge_details": {
            "faithfulness": {
                "reason": faithfulness_result["reason"],
                "parse_failed": faithfulness_result["parse_failed"],
            },
            "answer_relevance": {
                "reason": relevance_result["reason"],
                "parse_failed": relevance_result["parse_failed"],
            },
            "answer_correctness": (
                {
                    "reason": correctness_result["reason"],
                    "parse_failed": correctness_result["parse_failed"],
                }
                if correctness_result
                else None
            ),
        },
    }

# ---------------------------------------------------------------------------
# Statistical rigor: is a metric delta real, and do retrieval metrics
# actually predict generation quality?
# ---------------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, ci: float = 0.95, seed: int | None = None) -> dict:
    """Bootstrap confidence interval for the mean of `values`. With ~20
    eval queries, a bare average can look meaningfully different between
    two configs purely by which queries happened to be sampled -- this
    quantifies how much the mean could plausibly wobble on a re-sample of
    the same query set, so a metric can be reported as e.g.
    "0.72 [0.58, 0.85]" instead of a bare point estimate."""
    clean = np.array([v for v in values if v is not None], dtype=float)
    if len(clean) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "n_resamples": n_resamples}
    rng = np.random.default_rng(seed)
    n = len(clean)
    means = np.array([rng.choice(clean, size=n, replace=True).mean() for _ in range(n_resamples)])
    alpha = 1 - ci
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(clean.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": n,
        "n_resamples": n_resamples,
    }


def paired_significance_test(values_a: Sequence[float], values_b: Sequence[float]) -> dict:
    """Paired test on per-query (b - a) deltas -- appropriate here because
    two eval runs share the same eval_queries, so each query contributes a
    matched pair rather than two independent samples. Uses Wilcoxon
    signed-rank (nonparametric -- doesn't assume deltas are normally
    distributed, which they usually aren't for metrics bounded in [0,1]
    with small n) and falls back to a paired t-test if Wilcoxon can't run."""
    pairs = [(a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    if len(pairs) < 2:
        return {"test": None, "statistic": None, "p_value": None, "n": len(pairs)}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if np.allclose(a, b):
        return {"test": "none", "statistic": 0.0, "p_value": 1.0, "n": len(pairs)}
    try:
        stat, p = scipy_stats.wilcoxon(a, b)
        test_name = "wilcoxon"
    except ValueError:
        # Wilcoxon needs enough non-zero deltas; fall back for small/degenerate cases
        stat, p = scipy_stats.ttest_rel(a, b)
        test_name = "paired_t"
    return {"test": test_name, "statistic": float(stat), "p_value": float(p), "n": len(pairs)}


def correlation(values_a: Sequence[float], values_b: Sequence[float]) -> dict:
    """Spearman rank correlation (primary -- doesn't assume a linear
    relationship, which matters since hit_rate is binary and others are
    bounded ratios) plus Pearson, between two per-query metric series. Used
    to check whether a retrieval metric (e.g. recall) actually predicts a
    generation outcome (e.g. faithfulness) rather than assuming it does."""
    pairs = [(a, b) for a, b in zip(values_a, values_b) if a is not None and b is not None]
    if len(pairs) < 3:
        return {"spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None, "n": len(pairs)}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if np.std(a) == 0 or np.std(b) == 0:
        # scipy errors/NaNs on constant input (e.g. hit_rate all 1.0)
        return {"spearman_r": None, "spearman_p": None, "pearson_r": None, "pearson_p": None, "n": len(pairs)}
    sr, sp = scipy_stats.spearmanr(a, b)
    pr, pp = scipy_stats.pearsonr(a, b)
    return {
        "spearman_r": float(sr), "spearman_p": float(sp),
        "pearson_r": float(pr), "pearson_p": float(pp),
        "n": len(pairs),
    }