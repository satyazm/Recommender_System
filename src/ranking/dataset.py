"""Assemble the candidate pool with per-source retrieval metadata.

Each candidate carries WHICH strategy(ies) retrieved it and at what rank within
that strategy — the "recall strategy features" that top H&M solutions found
significantly improved ranking, since "retrieved by repurchase at rank 1" is a
very different signal from "retrieved by content-similarity at rank 40."
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from candidates.als import ALSRecommender
from candidates.itemitem import ItemItemRecommender
from candidates.popularity import PopularityRecommender
from candidates.recency import RecencyRecommender
from eval.metrics import build_ground_truth
from ranking.features import (
    category_affinity_features, item_popularity_features, item_price_features,
    user_features, user_item_history_features,
)


def build_candidate_pool(models: dict, user_idxs: list[int], k_per_source: int) -> pd.DataFrame:
    rows = {}  # (user_idx, item_idx) -> {f"src_{name}": bool, f"rank_{name}": int}
    for name, model in models.items():
        recs = model.recommend(user_idxs, k_per_source)
        for u, items in recs.items():
            for rank, item in enumerate(items):
                key = (u, item)
                if key not in rows:
                    rows[key] = {}
                rows[key][f"rank_{name}"] = rank

    all_source_names = list(models.keys())
    records = []
    for (u, item), source_ranks in rows.items():
        record = {"user_idx": u, "item_idx": item}
        for name in all_source_names:
            rank = source_ranks.get(f"rank_{name}")
            record[f"src_{name}"] = rank is not None
            record[f"rank_{name}"] = rank if rank is not None else k_per_source  # sentinel: "not retrieved"
        records.append(record)
    return pd.DataFrame.from_records(records)


def label_pool(pool: pd.DataFrame, ground_truth: dict[int, set[int]]) -> pd.DataFrame:
    pool = pool.copy()
    pool["label"] = [
        1 if item in ground_truth.get(u, set()) else 0
        for u, item in zip(pool["user_idx"], pool["item_idx"])
    ]
    return pool


def build_labeled_dataset(
    feature_df: pd.DataFrame, target_df: pd.DataFrame, target_date, user_map: pd.DataFrame,
    item_map: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame, k_per_source: int = 10,
) -> tuple[pd.DataFrame, dict, dict[int, set[int]]]:
    """feature_df must already be strictly before target_date. Fits fresh candidate-gen
    models on feature_df only, builds+labels the union candidate pool, and attaches
    every engineered feature. Returns (labeled_df, fitted_models) — models are returned
    so the same fit can be reused for the next call (e.g. val reuses what train touched)."""
    t0 = time.time()
    def _log(msg):
        print(f"  [{time.time() - t0:6.1f}s] {msg}", flush=True)

    models = {}
    for name, cls in [("recency", RecencyRecommender), ("itemitem", ItemItemRecommender),
                       ("als", ALSRecommender), ("popularity", PopularityRecommender)]:
        models[name] = cls().fit(feature_df)
        _log(f"fit {name}")

    # Restrict candidate generation to users active in the target window, not every
    # user in feature_df's whole history — feature_df spans up to 2 years, so "every
    # user who ever purchased" is ~1.3M people; generating + scoring candidates for
    # all of them is both unnecessary (we can only label users we have a target
    # outcome for) and, for ALS/item-item, prohibitively slow. This does mean the
    # training set has no "recently active user who churned entirely" negatives —
    # a real limitation worth naming, not a free simplification.
    if "user_idx" in target_df.columns:
        target_user_idxs = set(target_df["user_idx"])
    else:
        target_user_idxs = set(target_df.merge(user_map, on="customer_id")["user_idx"])
    user_idxs = list(set(feature_df["user_idx"]) & target_user_idxs)
    _log(f"target population = {len(user_idxs):,} users")
    pool = build_candidate_pool(models, user_idxs, k_per_source)
    _log(f"built candidate pool, {len(pool):,} rows")

    ground_truth = build_ground_truth(target_df, user_map, item_map)
    pool = label_pool(pool, ground_truth)
    _log("labeled pool")

    pool = user_item_history_features(feature_df, target_date, pool)
    _log("user_item_history_features")
    pool = category_affinity_features(feature_df, articles, pool)
    _log("category_affinity_features")
    pool = pool.merge(item_popularity_features(feature_df, target_date, item_map), on="item_idx", how="left")
    _log("item_popularity_features")
    pool = pool.merge(item_price_features(feature_df, item_map), on="item_idx", how="left")
    _log("item_price_features")
    pool = pool.merge(user_features(feature_df, target_date, user_map), on="user_idx", how="left")
    _log("user_features")
    pool = pool.merge(articles, on="item_idx", how="left")
    pool = pool.merge(customers, on="user_idx", how="left")
    _log("merged articles/customers")
    return pool, models, ground_truth
