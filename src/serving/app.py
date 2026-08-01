"""Phase 5/6: serving API. Exactly the 2 endpoints scoped — no auth, no monitoring.

/recommend/{customer_id} is a Redis lookup against recommendations precomputed
by train_ranker.py (the full candidate-gen + LightGBM pipeline is too slow to
run per-request), degrading gracefully for users the model never saw:
personalized -> age-bucket popularity (if we know their age) -> global
popularity. /similar_items/{article_id} runs FAISS live against ALS item
embeddings for warm articles, and against content embeddings (built from
product attributes alone, no purchase history needed) for the ~2,575 articles
that were never in the trained catalog — a true 404 only means the article_id
isn't in H&M's catalog at all.
"""
import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import faiss
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException

MODELS_DIR = "models"
SIM_CACHE_TTL_SECONDS = 3600
AGE_BUCKET_BINS = [0, 25, 35, 45, 55, 200]
AGE_BUCKET_LABELS = ["under_25", "25_34", "35_44", "45_54", "55_plus"]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["redis"] = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    state["faiss_index"] = faiss.read_index(f"{MODELS_DIR}/item_index.faiss")
    item_map = pd.read_parquet(f"{MODELS_DIR}/item_id_map_served.parquet")
    state["article_to_idx"] = dict(zip(item_map["article_id"], item_map["item_idx"]))
    state["idx_to_article"] = dict(zip(item_map["item_idx"], item_map["article_id"]))

    # Cold-item fallback: content embeddings cover EVERY article, including the
    # ~2,575 never in the ALS-trained catalog (new products, or items that only
    # ever transacted after our train cutoff).
    state["content_index"] = faiss.read_index(f"{MODELS_DIR}/content_index.faiss")
    all_articles = pd.read_parquet(f"{MODELS_DIR}/all_article_ids.parquet")
    state["content_idx_to_article"] = dict(enumerate(all_articles["article_id"]))
    state["content_article_to_idx"] = {v: k for k, v in state["content_idx_to_article"].items()}

    # Cold-user fallback: age-bucket popularity, used only when the precomputed
    # personalized list is missing for this customer_id.
    age_pop = pd.read_json(f"{MODELS_DIR}/age_bucket_popularity.json", typ="series")
    state["age_bucket_popularity"] = {bucket: [str(a) for a in items] for bucket, items in age_pop.items()}
    customer_age = pd.read_parquet(f"{MODELS_DIR}/customer_age.parquet")
    state["customer_age"] = dict(zip(customer_age["customer_id"], customer_age["age"]))

    yield
    state.clear()


app = FastAPI(title="H&M Recommender", lifespan=lifespan)


def _age_bucket(age: float) -> Optional[str]:
    if age is None or pd.isna(age):
        return None
    for lo, hi, label in zip(AGE_BUCKET_BINS, AGE_BUCKET_BINS[1:], AGE_BUCKET_LABELS):
        if lo <= age < hi:
            return label
    return None


@app.get("/recommend/{customer_id}")
def recommend(customer_id: str):
    cached = state["redis"].get(f"rec:{customer_id}")
    if cached is not None:
        return {"customer_id": customer_id, "recommendations": json.loads(cached), "fallback": None}

    age = state["customer_age"].get(customer_id)
    bucket = _age_bucket(age)
    if bucket is not None:
        return {
            "customer_id": customer_id,
            "recommendations": state["age_bucket_popularity"][bucket],
            "fallback": f"age_bucket:{bucket}",
        }

    fallback = state["redis"].get("popularity_fallback")
    if fallback is None:
        raise HTTPException(status_code=503, detail="Popularity fallback not loaded — run build_serving_artifacts.py")
    return {"customer_id": customer_id, "recommendations": json.loads(fallback), "fallback": "global_popularity"}


@app.get("/similar_items/{article_id}")
def similar_items(article_id: str, k: int = 12):
    article_id_int = int(article_id)
    item_idx = state["article_to_idx"].get(article_id_int)

    if item_idx is not None:
        cache_key = f"sim:{article_id}:{k}"
        cached = state["redis"].get(cache_key)
        if cached is not None:
            return {"article_id": article_id, "similar_items": json.loads(cached), "fallback": None}

        query = state["faiss_index"].reconstruct(int(item_idx)).reshape(1, -1)
        _scores, neighbor_idxs = state["faiss_index"].search(query, k + 1)  # own nearest neighbor is itself
        result = [str(state["idx_to_article"][i]) for i in neighbor_idxs[0] if i != item_idx][:k]
        state["redis"].set(cache_key, json.dumps(result), ex=SIM_CACHE_TTL_SECONDS)
        return {"article_id": article_id, "similar_items": result, "fallback": None}

    # Not in the ALS-trained catalog — try the content-based index instead, which
    # covers every article regardless of purchase history.
    content_idx = state["content_article_to_idx"].get(article_id_int)
    if content_idx is None:
        raise HTTPException(status_code=404, detail=f"article_id {article_id} not in H&M's catalog at all")

    query = state["content_index"].reconstruct(int(content_idx)).reshape(1, -1)
    _scores, neighbor_idxs = state["content_index"].search(query, k + 1)
    result = [str(state["content_idx_to_article"][i]) for i in neighbor_idxs[0] if i != content_idx][:k]
    return {"article_id": article_id, "similar_items": result, "fallback": "content_based (cold item)"}
