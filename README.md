# H&M Two-Stage Recommendation Engine

A production-style recommender built on real e-commerce interaction data (H&M Personalized Fashion Recommendations, Kaggle): candidate generation → learning-to-rank → ANN retrieval → served via FastAPI + Redis, with cold-start handling and an offline counterfactual A/B evaluation. Built to demonstrate recsys depth — modeling, evaluation, and the tradeoffs behind each design choice — rather than infra/orchestration depth.

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) was chosen over RetailRocket, Instacart, and Yoochoose/Diginetica for its combination of real purchase-behavior scale and rich product/customer metadata, after profiling all candidates for sparsity, event granularity, and timestamp quality.

| | |
|---|---|
| Transactions | 31,788,324 |
| Date range | 2018-09-20 → 2020-09-22 |
| Customers | 1,362,281 active (1,371,980 in catalog) |
| Articles | 104,547 active (105,542 in catalog) |
| Sparsity | 99.98% |
| Repeat-purchase rate | 14.1% of transactions are repeat buys |
| Cold articles (0 transactions ever) | 995 |
| Cold customers (0 transactions ever) | 9,699 |

The repeat-purchase rate and cold-entity counts directly motivated two design decisions below: a recency/repurchase candidate source, and a two-sided cold-start fallback strategy.

## Architecture

```
                    ┌─────────────────────┐
                    │  Candidate Generation │  (4 sources, union pool)
                    │  popularity · item-item co-occurrence
                    │  ALS (collaborative filtering) · content (TF-IDF+SVD)
                    └──────────┬───────────┘
                               │ candidates + per-source rank features
                    ┌──────────▼───────────┐
                    │   LightGBM Ranker     │  (binary classification,
                    │  (feature engineering) │   time-based train/val/test)
                    └──────────┬───────────┘
                               │ top-12 per user (precomputed, offline)
              ┌────────────────┼────────────────┐
              │                                 │
     ┌────────▼────────┐              ┌─────────▼─────────┐
     │  Redis (cache)   │              │  FAISS IndexFlatIP │  (live, <1ms)
     │  /recommend      │              │  /similar_items    │
     └──────────────────┘              └────────────────────┘
              │                                 │
       cold-user fallback:              cold-item fallback:
       age-bucket → global popularity   content-based embedding index
```

## Time-based split

Matches the actual competition protocol: last 7 days = test, prior 7 days = val, everything before = train. A **time-based split is not optional here** — a random split would let the model implicitly see the future relative to what it's asked to predict, since a customer's later purchase could land in train while an earlier one lands in test.

The ranker's own training set is built the same way, shifted one week earlier (train-minus-last-week → last-train-week as labels), mirroring "train on historical weeks, validate on the immediately following week." **Test is touched exactly once**, in Phase 7, after every other decision was already finalized.

## Results

### Candidate generation (Recall@K, val set, warm users)

| method | Recall@10 | Recall@20 | Recall@50 |
|---|---|---|---|
| popularity (baseline) | 0.73% | 1.61% | 2.74% |
| item-item co-occurrence | 2.39% | 3.65% | 6.11% |
| ALS | 1.97% | 2.85% | 4.37% |
| content-based (TF-IDF+SVD) | 0.48% | 0.71% | 1.25% |
| recency/repurchase | 4.40% | 4.86% | 5.31% |
| **union of all 4** | **6.40%** | **8.06%** | **10.39%** |

Recency/repurchase is the single strongest source — H&M's 14.1% repeat-purchase rate isn't a curiosity, it's the dominant signal, and it's cheap: no model, just a sort. The union beats any single method because different generators retrieve genuinely different candidates, not because any one of them is individually best.

### Final ranking (test set, touched once)

| method | Recall@12 | NDCG@12 | MAP@12 |
|---|---|---|---|
| popularity | 0.80% | 0.54% | 0.30% |
| heuristic (best source rank, no ML) | 4.49% | 3.25% | 2.29% |
| **LightGBM ranker** | **5.60%** | **4.18%** | **2.96%** |

**LightGBM beats popularity by 7-10x across all three metrics**, and adds +29% NDCG@12 / +30% MAP@12 over a non-ML heuristic ordering of the *same* candidate pool — isolating what the ranking stage itself contributes, separate from candidate quality. MAP@12 = 0.0296 sits in the neighborhood of the actual competition's winning leaderboard scores, with a single untuned LightGBM model and no ensemble. Val and test scores are nearly identical (MAP@12 = 0.0296 both), indicating the model generalized rather than overfitting to val.

Top ranking features: `department_name`, `colour_group_name`, `product_type_name` (item content), then `user_avg_price`, `days_since_last_purchase`, `age`, `days_since_last_purchase_of_item`, `type_affinity`, `user_repeat_rate` — user-item interaction and demographic features dominate over pure popularity signals.

### FAISS index choice (Phase 4)

Benchmarked Flat/IVF/HNSW on the actual 103K-item, 64-dim ALS embedding catalog rather than assuming an index:

| Index | Build time | Query latency | Recall@10 |
|---|---|---|---|
| **Flat (exact)** | instant | **0.035 ms** | **1.0000** |
| IVF (nlist=1283, nprobe=64) | 1.3s | 0.008 ms | 0.898 |
| HNSW (M=32, efSearch=256) | 21.9s | 0.040 ms | 0.683 |

**Exact search wins outright at this scale.** Flat's cost is O(N·d) per query; with N=103K, d=64 that's already sub-millisecond, so there's no recall/latency tradeoff left to make — IVF and HNSW only pay off once the linear scan itself becomes the bottleneck, roughly a catalog in the millions-to-tens-of-millions range. Choosing `IndexFlatIP` here is the conclusion of the tradeoff analysis, not a shortcut around it. (Also: ALS ranks by inner product, not Euclidean distance — using `IndexFlatL2` would silently optimize the wrong metric.)

### Offline counterfactual evaluation (Phase 7)

H&M's transaction log has no recorded recommendation propensities (it's a purchase log, not a bandit log), so **Inverse Propensity Scoring (IPS)** was chosen over a synthetic click simulator — a real off-policy evaluation technique (used at Netflix/Spotify to estimate a new policy's performance before deploying it), applied honestly with a stated logging-policy assumption rather than a circular synthetic demo.

| method | uniform propensity (SNIPS) | popularity propensity (SNIPS) |
|---|---|---|
| popularity | 0.82% | **0.00%** |
| heuristic | 3.47% | 0.89% |
| LightGBM ranker | 4.60% | 1.13% |

Under uniform propensity, self-normalized IPS (SNIPS) mathematically collapses to the plain unweighted hit rate — a built-in sanity check that the estimator is correct, not a coincidence. Under the more realistic popularity-proportional propensity, **the popularity baseline's score drops to ~0**: its naive success was almost entirely popularity bias (it's trivially easy to get credit for recommending items everyone buys anyway). LightGBM's edge over the heuristic baseline, in contrast, holds at a stable ~25-27% whether measured naively or bias-corrected — evidence of genuine personalization skill, not an artifact of the same bias.

Caveats stated explicitly, not hidden: the log contains only positive (purchase) events, no observed negatives; and the propensity itself is an assumption, not a logged ground truth.

## Cold-start handling (Phase 6)

Two independent gaps, closed with the cheapest fix that's still honest:

- **New items**: 2,575 articles were never in the ALS-trained catalog (either truly zero-transaction, or first appearing after the train cutoff). Content embeddings (TF-IDF + SVD over product attributes) need no purchase history, so `/similar_items` falls back to a content-based FAISS index for these — a true 404 now only means the article isn't in H&M's catalog at all.
- **New users**: `/recommend` degrades personalized → age-bucket popularity (if we know the customer's age, even with zero transactions) → global popularity (last resort). Each tier only degrades as much as the available signal requires.

## Serving (Phase 5)

Two endpoints, no auth, no monitoring dashboard — deliberately minimal, since this project's other half is an MLOps-focused repo already covering that ground.

- **`/recommend/{customer_id}`**: the full candidate-gen + LightGBM pipeline takes ~18 minutes to run, far too slow per-request — so recommendations are precomputed offline and Redis serves the lookup (a batch-score/cache-serve pattern, not a live model call).
- **`/similar_items/{article_id}`**: FAISS exact search runs live per request (0.035ms, per the Phase 4 benchmark), cached in Redis for repeat lookups.

## Project structure

```
scripts/
  profile_hm.py, profile_retailrocket.py   Phase 1: dataset profiling
  build_split.py                           time-based train/val/test split + id maps
  eval_phase2_candidates.py                candidate-gen comparison (Phase 2)
  train_ranker.py                          feature engineering + LightGBM (Phase 3)
  benchmark_faiss.py                       Flat/IVF/HNSW benchmark (Phase 4)
  build_serving_artifacts.py               push precomputed recs to Redis, build FAISS index
  build_cold_start_artifacts.py            age-bucket popularity + full-catalog content index
  evaluate_test_ips.py                     final test eval + IPS/SNIPS (Phase 7)

src/
  candidates/    popularity, itemitem, als, content, recency recommenders
  ranking/       feature engineering + labeled dataset construction
  eval/          Recall@K, NDCG@K, MAP@K
  serving/       FastAPI app

models/          persisted artifacts (LightGBM model, ALS embeddings, FAISS indices, precomputed recs)
data/processed/  parquet transactions + id mappings
Dockerfile, requirements-serving.txt   lean serving image (no training deps)
```

## Running it

```bash
# Data prep + training (one-time, ~18 min — ALS fitting dominates)
python3 scripts/build_split.py
python3 scripts/train_ranker.py
python3 scripts/build_serving_artifacts.py
python3 scripts/build_cold_start_artifacts.py

# Serving
docker network create recsys-net
docker run -d --name recsys-redis --network recsys-net -p 6379:6379 redis:7-alpine
docker build -t hm-recommender-api .
docker run -d --name recsys-api --network recsys-net -p 8000:8000 -e REDIS_HOST=recsys-redis hm-recommender-api
```

API docs at `http://127.0.0.1:8000/docs` once running.

```bash
# Final held-out test evaluation + IPS counterfactual analysis (run once, after everything else)
python3 scripts/evaluate_test_ips.py
```

## What was deliberately kept minimal

Per the project's own scope: Airflow (no DAG — training is a sequence of scripts), Docker (one lean Dockerfile for the serving image, not the training environment), Redis (a cache in front of precomputed output, not a feature store), and no model ensembling (a single LightGBM matches what top competition solutions found was the strongest *individual* model — ensembling is marginal gains for real complexity cost, not worth defending here). These were all conscious tradeoffs, not omissions.
