"""Phase 7: final test evaluation (touched exactly once) + IPS/SNIPS offline
counterfactual evaluation.

Candidate-gen models are refit on train+val (everything known by the time we'd
actually serve test-period users) against FRESH user/item id maps built from
train+val — reusing the original train-only maps would silently drop any item
that first appeared during val, before it ever reached the candidate generators.

The persisted LightGBM model, however, was fit with categorical codes derived
from the train-only item subset (which is missing 2 real category values —
'Dog wear', 'Pre-walkers' — entirely absent from train). Categorical columns
here are built against that EXACT reference category list, not re-inferred
fresh, so any newly-appearing category value becomes NaN (a legitimate
"missing" signal to LightGBM) instead of silently colliding with a different
learned code.

IPS/SNIPS: H&M's transaction log has no recorded recommendation propensities
(it's a purchase log, not a bandit log), so a logging-policy propensity has to
be assumed. Two are computed for a sensitivity check: uniform-over-catalog
(the weakest, most conservative assumption) and popularity-proportional
(treating observed purchase frequency as a proxy for how likely an item was to
be surfaced at all — the more standard simplifying assumption in the
literature when true logged propensities don't exist). Self-normalized IPS
(SNIPS) is used instead of raw IPS because propensities this small (some
1/100K+) produce enormous, high-variance importance weights under raw IPS.
"""
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from eval.metrics import map_at_k, ndcg_at_k, recall_at_k
from ranking.dataset import build_labeled_dataset

DATA_DIR = "data/processed"
K_PER_SOURCE = 10
TOP_N = 12

txns = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")

# Fresh maps from train+val — see module docstring for why the original
# train-only maps would be wrong here.
train_val_txns = txns[txns["split"].isin(["train", "val"])]
user_map = pd.DataFrame({"customer_id": train_val_txns["customer_id"].unique()})
user_map["user_idx"] = range(len(user_map))
item_map = pd.DataFrame({"article_id": train_val_txns["article_id"].unique()})
item_map["item_idx"] = range(len(item_map))

ARTICLE_CATEGORICAL = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "department_name", "index_name", "section_name", "garment_group_name",
]
CUSTOMER_CATEGORICAL = ["club_member_status", "fashion_news_frequency"]

# Reference categories, derived EXACTLY as train_ranker.py did (train-only item/user
# subset) so codes line up with what the persisted model actually learned.
orig_item_map = pd.read_parquet(f"{DATA_DIR}/item_id_map.parquet")
orig_user_map = pd.read_parquet(f"{DATA_DIR}/user_id_map.parquet")

articles_raw = pd.read_csv("data/raw/hm/articles.csv", dtype={"article_id": "int32"})
ref_articles = articles_raw.merge(orig_item_map, on="article_id")
reference_article_categories = {col: pd.Index(ref_articles[col].astype("category").cat.categories)
                                 for col in ARTICLE_CATEGORICAL}

customers_raw = pd.read_csv("data/raw/hm/customers.csv", dtype={"customer_id": "category"})
ref_customers = customers_raw.merge(orig_user_map, on="customer_id")
reference_customer_categories = {col: pd.Index(ref_customers[col].fillna("UNKNOWN").astype("category").cat.categories)
                                  for col in CUSTOMER_CATEGORICAL}

articles = articles_raw.merge(item_map, on="article_id").drop(columns=["article_id"])
articles = articles[["item_idx"] + ARTICLE_CATEGORICAL]
for col in ARTICLE_CATEGORICAL:
    articles[col] = pd.Categorical(articles[col], categories=reference_article_categories[col])

customers = customers_raw.merge(user_map, on="customer_id").drop(columns=["customer_id"])
customers = customers[["user_idx", "age", "FN", "Active"] + CUSTOMER_CATEGORICAL]
customers["FN"] = customers["FN"].fillna(0)
customers["Active"] = customers["Active"].fillna(0)
for col in CUSTOMER_CATEGORICAL:
    customers[col] = pd.Categorical(customers[col].fillna("UNKNOWN"), categories=reference_customer_categories[col])

train_plus_val = train_val_txns.merge(user_map, on="customer_id").merge(item_map, on="article_id")
test_target = txns[txns["split"] == "test"]
test_start = train_plus_val["t_dat"].max() + pd.Timedelta(days=1)

print(f"Building test candidate pool (features < {test_start.date()}, labels = test week)...")
test_pool, test_models, test_ground_truth = build_labeled_dataset(
    train_plus_val, test_target, test_start, user_map, item_map, articles, customers, k_per_source=K_PER_SOURCE,
)
print(f"  {len(test_pool):,} rows, {test_pool['label'].sum():,} positive ({test_pool['label'].mean():.4%})")

saved = joblib.load("models/lightgbm_ranker.joblib")
model, feature_cols, categorical_cols = saved["model"], saved["feature_cols"], saved["categorical_cols"]
test_pool["score"] = model.predict_proba(test_pool[feature_cols])[:, 1]

ranked = test_pool.sort_values(["user_idx", "score"], ascending=[True, False])
lgbm_recs = ranked.groupby("user_idx")["item_idx"].apply(lambda s: list(s)[:TOP_N]).to_dict()

best_rank_cols = [c for c in feature_cols if c.startswith("rank_")]
test_pool["best_rank"] = test_pool[best_rank_cols].min(axis=1)
heuristic = test_pool.sort_values(["user_idx", "best_rank"], ascending=[True, True])
heuristic_recs = heuristic.groupby("user_idx")["item_idx"].apply(lambda s: list(s)[:TOP_N]).to_dict()

pop_users = list(test_ground_truth.keys())
popularity_recs = test_models["popularity"].recommend(pop_users, TOP_N)

print(f"\n=== FINAL TEST-SET RESULT (touched once) ===")
print(f"{'method':<20}{'Recall@' + str(TOP_N):<15}{'NDCG@' + str(TOP_N):<15}{'MAP@' + str(TOP_N):<15}")
policies = {"popularity": popularity_recs, "heuristic": heuristic_recs, "LightGBM ranker": lgbm_recs}
for name, recs in policies.items():
    r = recall_at_k(recs, test_ground_truth, TOP_N)
    n = ndcg_at_k(recs, test_ground_truth, TOP_N)
    m = map_at_k(recs, test_ground_truth, TOP_N)
    print(f"{name:<20}{r:<15.4f}{n:<15.4f}{m:<15.4f}")

# --- IPS / SNIPS offline counterfactual evaluation ---
item_pop_counts = train_plus_val.groupby("item_idx").size()
n_items = item_map.shape[0]
total_purchases = item_pop_counts.sum()
pop_propensity = (item_pop_counts + 1) / (total_purchases + n_items)  # Laplace-smoothed
uniform_propensity = 1.0 / n_items

logged_events = test_target.merge(user_map, on="customer_id").merge(item_map, on="article_id")
logged_events = logged_events[logged_events["user_idx"].isin(pop_users)]


def snips(recs: dict, propensity: pd.Series) -> float:
    weights = logged_events["item_idx"].map(propensity).fillna(propensity.min()).to_numpy()
    ips_weights = 1.0 / weights
    hits = np.array([
        1.0 if item in recs.get(u, []) else 0.0
        for u, item in zip(logged_events["user_idx"], logged_events["item_idx"])
    ])
    return (ips_weights * hits).sum() / ips_weights.sum()


print(f"\n=== SNIPS-weighted hit rate (offline counterfactual estimate) ===")
print(f"{'method':<20}{'uniform propensity':<22}{'popularity propensity':<22}")
for name, recs in policies.items():
    snips_uniform = snips(recs, pd.Series(uniform_propensity, index=item_map["item_idx"]))
    snips_pop = snips(recs, pop_propensity)
    print(f"{name:<20}{snips_uniform:<22.4f}{snips_pop:<22.4f}")
