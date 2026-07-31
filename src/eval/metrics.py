"""Evaluation metrics.

Recall@K is used for candidate generation (Phase 2) — it only asks whether
relevant items are somewhere in the top-K set. NDCG@K and MAP@K are used for
the ranker (Phase 3) — they additionally reward putting relevant items EARLIER
in the top-K, which is what actually matters once a small, already-relevant
candidate set needs to be ordered into a top-12 shown to the user.
"""
import math
from collections import defaultdict

import pandas as pd


def build_ground_truth(eval_df: pd.DataFrame, user_map: pd.DataFrame, item_map: pd.DataFrame) -> dict[int, set[int]]:
    """user_idx -> set of item_idx purchased in the eval window.

    Restricted to (user, item) pairs where BOTH were seen in train. A ground-truth
    item that never appeared in train cannot possibly be retrieved by any method
    built on train data — including it would make every method's Recall@K
    artificially, and misleadingly, look worse than it actually is.
    """
    merged = eval_df
    if "user_idx" not in merged.columns:
        merged = merged.merge(user_map, on="customer_id", how="inner")
    if "item_idx" not in merged.columns:
        merged = merged.merge(item_map, on="article_id", how="inner")
    else:
        merged = merged[merged["item_idx"].isin(item_map["item_idx"])]
    merged = merged[merged["user_idx"].isin(user_map["user_idx"])]
    ground_truth = defaultdict(set)
    for user_idx, item_idx in zip(merged["user_idx"], merged["item_idx"]):
        ground_truth[user_idx].add(item_idx)
    return dict(ground_truth)


def recall_at_k(recommended: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int) -> float:
    """Mean over users of |top-k recommended ∩ relevant| / |relevant|."""
    scores = []
    for user_idx, relevant in ground_truth.items():
        if not relevant:
            continue
        topk = set(recommended.get(user_idx, [])[:k])
        scores.append(len(topk & relevant) / len(relevant))
    return sum(scores) / len(scores) if scores else 0.0


def ndcg_at_k(recommended: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int) -> float:
    scores = []
    for user_idx, relevant in ground_truth.items():
        if not relevant:
            continue
        topk = recommended.get(user_idx, [])[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(topk) if item in relevant)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def map_at_k(recommended: dict[int, list[int]], ground_truth: dict[int, set[int]], k: int) -> float:
    scores = []
    for user_idx, relevant in ground_truth.items():
        if not relevant:
            continue
        topk = recommended.get(user_idx, [])[:k]
        hits = 0
        precision_sum = 0.0
        for i, item in enumerate(topk):
            if item in relevant:
                hits += 1
                precision_sum += hits / (i + 1)
        scores.append(precision_sum / min(len(relevant), k))
    return sum(scores) / len(scores) if scores else 0.0
