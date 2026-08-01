"""Phase 4: FAISS index selection, justified empirically at OUR actual scale
(103K items, 64-dim ALS item embeddings) — not assumed from general ANN lore.

ALS ranks items by inner product (not Euclidean distance), so IndexFlatIP is the
correct "exact" baseline here — using L2 would optimize the wrong objective and
make every comparison meaningless from the start.

Flat = exact brute-force search (the ground truth every approximate method is
graded against). IVF clusters vectors into nlist cells and only searches nprobe
of them, trading recall for speed. HNSW builds a navigable graph, trading index
memory/build-time for very fast, usually higher-recall queries than IVF at
comparable speed. Which one wins depends entirely on catalog size and latency
budget — that's exactly what we measure below instead of assuming.
"""
import time

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

DATA_DIR = "data/processed"

txns = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")
user_map = pd.read_parquet(f"{DATA_DIR}/user_id_map.parquet")
item_map = pd.read_parquet(f"{DATA_DIR}/item_id_map.parquet")
train = txns[txns["split"] == "train"].merge(user_map, on="customer_id").merge(item_map, on="article_id")

n_users = train["user_idx"].max() + 1
n_items = train["item_idx"].max() + 1
counts = train.groupby(["user_idx", "item_idx"]).size().reset_index(name="count")
user_items = sp.csr_matrix(
    (counts["count"].astype(np.float32), (counts["user_idx"], counts["item_idx"])), shape=(n_users, n_items)
)

print("Fitting ALS for item embeddings...")
model = AlternatingLeastSquares(factors=64, regularization=0.01, alpha=15.0, iterations=15, random_state=42)
model.fit(user_items, show_progress=False)
item_vectors = np.ascontiguousarray(model.item_factors.astype(np.float32))
user_vectors = np.ascontiguousarray(model.user_factors.astype(np.float32))
d = item_vectors.shape[1]
print(f"Item catalog: {n_items:,} items, {d}-dim embeddings "
      f"({item_vectors.nbytes / 1e6:.1f}MB raw)")

# --- ground truth: exact search ---
flat = faiss.IndexFlatIP(d)
flat.add(item_vectors)

n_queries = 2000
rng = np.random.default_rng(42)
query_idx = rng.choice(n_users, size=n_queries, replace=False)
queries = np.ascontiguousarray(user_vectors[query_idx])
K = 10

t0 = time.time()
gt_scores, gt_ids = flat.search(queries, K)
flat_query_time = (time.time() - t0) / n_queries * 1000
print(f"\nFlat (exact):        build instant, {flat_query_time:.4f} ms/query (batch of {n_queries}), "
      f"recall@{K} = 1.0000 (this IS ground truth)")


def recall_vs_gt(result_ids: np.ndarray) -> float:
    hits = sum(len(set(result_ids[i]) & set(gt_ids[i])) for i in range(n_queries))
    return hits / (n_queries * K)


# --- IVF: cluster into nlist cells, search nprobe of them ---
nlist = int(4 * np.sqrt(n_items))  # standard heuristic: a few dozen vectors per cell on average
quantizer = faiss.IndexFlatIP(d)
ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
t0 = time.time()
ivf.train(item_vectors)
ivf.add(item_vectors)
ivf_build_time = time.time() - t0

print(f"\nIVF (nlist={nlist}): build {ivf_build_time:.2f}s")
for nprobe in [1, 4, 16, 64]:
    ivf.nprobe = nprobe
    t0 = time.time()
    _scores, ids = ivf.search(queries, K)
    query_time = (time.time() - t0) / n_queries * 1000
    print(f"  nprobe={nprobe:<4} {query_time:.4f} ms/query   recall@{K} = {recall_vs_gt(ids):.4f}")

# --- HNSW: navigable small-world graph ---
for M in [16, 32]:
    hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    hnsw.add(item_vectors)
    hnsw_build_time = time.time() - t0
    print(f"\nHNSW (M={M}): build {hnsw_build_time:.2f}s, "
          f"index size ~{(item_vectors.nbytes + hnsw.hnsw.entry_point * 0 + M * n_items * 4 * 2) / 1e6:.1f}MB (graph overhead)")
    for ef in [16, 64, 256]:
        hnsw.hnsw.efSearch = ef
        t0 = time.time()
        _scores, ids = hnsw.search(queries, K)
        query_time = (time.time() - t0) / n_queries * 1000
        print(f"  efSearch={ef:<4} {query_time:.4f} ms/query   recall@{K} = {recall_vs_gt(ids):.4f}")
