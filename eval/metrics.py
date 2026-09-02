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


def _judge_score(prompt: str, label: str) -> float:
    print(f"    judging {label}...", flush=True)
    llm = _get_judge_llm()
    response = llm.chat([ChatMessage(role="user", content=prompt)])
    data = extract_json(str(response))
    if data is None:
        print(f"    judging {label}: no parseable JSON in judge response — scoring 0.0", flush=True)
        return 0.0
    try:
        score = float(data.get("score", 0.0))
    except (ValueError, TypeError):
        print(f"    judging {label}: malformed JSON in judge response — scoring 0.0", flush=True)
        return 0.0
    print(f"    judging {label}: {score}", flush=True)
    return score


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
    return _judge_score(prompt, label="faithfulness")


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
    return _judge_score(prompt, label="answer_relevance")


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
    return _judge_score(prompt, label="answer_correctness")


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