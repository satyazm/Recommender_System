"""Phase 5, step 1: push precomputed recommendations into Redis, build the FAISS
item index for live /similar_items lookups.

Two different serving strategies for the two endpoints, deliberately:
- /recommend needs the full candidate-gen + LightGBM pipeline, which takes ~18
  minutes to run (see train_ranker.py) — far too slow for a request. So it's
  precomputed offline and Redis just serves the lookup.
- /similar_items only needs a nearest-neighbor search over item embeddings,
  which Phase 4 measured at 0.035ms with exact search — fast enough to run
  live per request, no precomputation needed. Redis caches it anyway (a
  cache-aside layer for repeat lookups), but isn't load-bearing here the way
  it is for /recommend.
"""
import json

import faiss
import numpy as np
import pandas as pd
import redis

MODELS_DIR = "models"
POP_FALLBACK_TTL = None  # refreshed by the next training run, not time-based
REC_TTL_SECONDS = 7 * 24 * 3600

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

item_map = pd.read_parquet(f"{MODELS_DIR}/item_id_map_served.parquet")
item_idx_to_article = item_map.set_index("item_idx")["article_id"].to_dict()

# --- push precomputed /recommend results ---
precomputed = pd.read_parquet(f"{MODELS_DIR}/precomputed_recs.parquet")
pipe = r.pipeline()
for customer_id, item_idxs in zip(precomputed["customer_id"], precomputed["item_idxs"]):
    article_ids = [str(item_idx_to_article[i]) for i in item_idxs]
    pipe.set(f"rec:{customer_id}", json.dumps(article_ids), ex=REC_TTL_SECONDS)
pipe.execute()
print(f"Pushed {len(precomputed):,} precomputed user recommendations to Redis.")

popularity_fallback = pd.read_parquet(f"{MODELS_DIR}/popularity_fallback.parquet")
r.set("popularity_fallback", json.dumps([str(a) for a in popularity_fallback["article_id"]]))
print(f"Pushed popularity fallback ({len(popularity_fallback)} items) to Redis.")

# --- build + save the FAISS index for /similar_items ---
item_vectors = np.load(f"{MODELS_DIR}/als_item_factors.npy")
index = faiss.IndexFlatIP(item_vectors.shape[1])
index.add(item_vectors)
faiss.write_index(index, f"{MODELS_DIR}/item_index.faiss")
print(f"Built + saved FAISS Flat index: {index.ntotal:,} items, dim={item_vectors.shape[1]}.")
