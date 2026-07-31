"""Phase 2 offline eval harness — run each candidate-gen method against the val
split and report Recall@K. Val (not test) is used here deliberately: test stays
untouched until we've picked a final method, otherwise we'd be tuning against
the same set we report a final number on.
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
from candidates.als import ALSRecommender
from candidates.content import ContentRecommender
from candidates.itemitem import ItemItemRecommender
from candidates.popularity import PopularityRecommender
from candidates.recency import RecencyRecommender
from eval.metrics import build_ground_truth, recall_at_k

DATA_DIR = "data/processed"
K_VALUES = [10, 20, 50]

txns = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")
user_map = pd.read_parquet(f"{DATA_DIR}/user_id_map.parquet")
item_map = pd.read_parquet(f"{DATA_DIR}/item_id_map.parquet")

train = txns[txns["split"] == "train"].merge(user_map, on="customer_id").merge(item_map, on="article_id")
val = txns[txns["split"] == "val"]

ground_truth = build_ground_truth(val, user_map, item_map)
print(f"Warm val users (in train, used for eval): {len(ground_truth):,}")

results = {}

pop_model = PopularityRecommender().fit(train)
pop_recs = pop_model.recommend(list(ground_truth.keys()), k=max(K_VALUES))
results["popularity"] = {k: recall_at_k(pop_recs, ground_truth, k) for k in K_VALUES}

print("Fitting item-item co-occurrence...")
ii_model = ItemItemRecommender().fit(train)
ii_recs = ii_model.recommend(list(ground_truth.keys()), k=max(K_VALUES))
results["item-item"] = {k: recall_at_k(ii_recs, ground_truth, k) for k in K_VALUES}

print("Fitting ALS...")
als_model = ALSRecommender().fit(train)
als_recs = als_model.recommend(list(ground_truth.keys()), k=max(K_VALUES))
results["als"] = {k: recall_at_k(als_recs, ground_truth, k) for k in K_VALUES}

print("Fitting content-based (TF-IDF + SVD)...")
content_model = ContentRecommender("data/raw/hm/articles.csv", item_map).fit(train)
content_recs = content_model.recommend(list(ground_truth.keys()), k=max(K_VALUES))
results["content"] = {k: recall_at_k(content_recs, ground_truth, k) for k in K_VALUES}

print("Fitting recency/repurchase...")
recency_model = RecencyRecommender().fit(train)
recency_recs = recency_model.recommend(list(ground_truth.keys()), k=max(K_VALUES))
results["recency"] = {k: recall_at_k(recency_recs, ground_truth, k) for k in K_VALUES}

# Union recall: at retrieval depth K, take the top-K from EACH source and dedupe —
# pool size is up to 4*K, not K. This is the real question a candidate-gen stage
# answers: "if each strategy contributes its top-K, what fraction of test purchases
# does the combined pool cover, before ranking even happens?"
union_scores = {}
for k in K_VALUES:
    union_recs = {
        u: list(dict.fromkeys(
            recency_recs.get(u, [])[:k] + ii_recs.get(u, [])[:k]
            + als_recs.get(u, [])[:k] + pop_recs.get(u, [])[:k]
        ))
        for u in ground_truth
    }
    union_scores[k] = recall_at_k(union_recs, ground_truth, k=10**9)
results["union (4x K pool)"] = union_scores

print(f"\n{'method':<15}" + "".join(f"Recall@{k:<10}" for k in K_VALUES))
for method, scores in results.items():
    print(f"{method:<15}" + "".join(f"{scores[k]:<17.4f}" for k in K_VALUES))
