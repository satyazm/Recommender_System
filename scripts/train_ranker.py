"""Phase 3: build leakage-safe ranker train/val sets, train LightGBM, evaluate.

Two labeled datasets are built with the SAME recipe, shifted one week apart:
  - ranker-train: features from train-minus-last-week, labels = last train week
  - ranker-val:   features from all of train,           labels = val week
This mirrors the "train on historical weeks, validate on the immediately
following week" protocol used by the actual H&M competition's top solutions.
Test stays untouched until a final model is chosen.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from eval.metrics import map_at_k, ndcg_at_k, recall_at_k
from ranking.dataset import build_labeled_dataset

DATA_DIR = "data/processed"
K_PER_SOURCE = 10
TOP_N = 12  # H&M's actual submission format: 12 recommendations per customer

txns = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")
user_map = pd.read_parquet(f"{DATA_DIR}/user_id_map.parquet")
item_map = pd.read_parquet(f"{DATA_DIR}/item_id_map.parquet")

ARTICLE_CATEGORICAL = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "department_name", "index_name", "section_name", "garment_group_name",
]
articles = pd.read_csv("data/raw/hm/articles.csv", dtype={"article_id": "int32"})
articles = articles.merge(item_map, on="article_id").drop(columns=["article_id"])
articles = articles[["item_idx"] + ARTICLE_CATEGORICAL]
for col in ARTICLE_CATEGORICAL:
    articles[col] = articles[col].astype("category")

CUSTOMER_CATEGORICAL = ["club_member_status", "fashion_news_frequency"]
customers = pd.read_csv("data/raw/hm/customers.csv", dtype={"customer_id": "category"})
customers = customers.merge(user_map, on="customer_id").drop(columns=["customer_id"])
customers = customers[["user_idx", "age", "FN", "Active"] + CUSTOMER_CATEGORICAL]
customers["FN"] = customers["FN"].fillna(0)
customers["Active"] = customers["Active"].fillna(0)
for col in CUSTOMER_CATEGORICAL:
    customers[col] = customers[col].fillna("UNKNOWN").astype("category")

# Full train, needed to slice the two feature/target windows below.
full_train = txns[txns["split"] == "train"].merge(user_map, on="customer_id").merge(item_map, on="article_id")
val_target = txns[txns["split"] == "val"]

val_start = full_train["t_dat"].max() + pd.Timedelta(days=1)  # first day NOT in train == val start
ranker_train_target_start = val_start - pd.Timedelta(days=7)

train_feature_df = full_train[full_train["t_dat"] < ranker_train_target_start]
train_target_df = full_train[full_train["t_dat"] >= ranker_train_target_start]

print(f"Ranker-train: features < {ranker_train_target_start.date()}, "
      f"labels {ranker_train_target_start.date()} .. {(val_start - pd.Timedelta(days=1)).date()}")
print(f"Ranker-val:   features < {val_start.date()}, labels = val week")

print("\nBuilding ranker-train set...")
train_pool, _train_models, _ = build_labeled_dataset(
    train_feature_df, train_target_df, ranker_train_target_start, user_map, item_map,
    articles, customers, k_per_source=K_PER_SOURCE,
)
print(f"  {len(train_pool):,} rows, {train_pool['label'].sum():,} positive ({train_pool['label'].mean():.4%})")

print("Building ranker-val set...")
val_pool, val_models, val_ground_truth = build_labeled_dataset(
    full_train, val_target, val_start, user_map, item_map, articles, customers, k_per_source=K_PER_SOURCE,
)
print(f"  {len(val_pool):,} rows, {val_pool['label'].sum():,} positive ({val_pool['label'].mean():.4%})")

DROP_COLS = ["user_idx", "item_idx", "label"]
categorical_cols = ARTICLE_CATEGORICAL + CUSTOMER_CATEGORICAL
feature_cols = [c for c in train_pool.columns if c not in DROP_COLS]

model = lgb.LGBMClassifier(
    objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, is_unbalance=True, random_state=42, verbose=-1,
)
model.fit(
    train_pool[feature_cols], train_pool["label"],
    categorical_feature=categorical_cols,
    eval_set=[(val_pool[feature_cols], val_pool["label"])],
    callbacks=[lgb.early_stopping(30, verbose=False)],
)

val_pool["score"] = model.predict_proba(val_pool[feature_cols])[:, 1]

# Ranked-by-model recommendations
ranked = val_pool.sort_values(["user_idx", "score"], ascending=[True, False])
ranker_recs = ranked.groupby("user_idx")["item_idx"].apply(list).to_dict()

# "No ranker" baseline: keep the union pool but order candidates by best (lowest) rank
# across sources instead of a learned model — isolates what the LightGBM stage adds
# on top of candidate generation alone.
best_rank_cols = [c for c in feature_cols if c.startswith("rank_")]
val_pool["best_rank"] = val_pool[best_rank_cols].min(axis=1)
heuristic = val_pool.sort_values(["user_idx", "best_rank"], ascending=[True, True])
heuristic_recs = heuristic.groupby("user_idx")["item_idx"].apply(list).to_dict()

print(f"\n{'method':<30}{'Recall@' + str(TOP_N):<15}{'NDCG@' + str(TOP_N):<15}{'MAP@' + str(TOP_N):<15}")
for name, recs in [("heuristic (best source rank)", heuristic_recs), ("LightGBM ranker", ranker_recs)]:
    r = recall_at_k(recs, val_ground_truth, TOP_N)
    n = ndcg_at_k(recs, val_ground_truth, TOP_N)
    m = map_at_k(recs, val_ground_truth, TOP_N)
    print(f"{name:<30}{r:<15.4f}{n:<15.4f}{m:<15.4f}")

print("\nTop 15 feature importances:")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.head(15).to_string())
