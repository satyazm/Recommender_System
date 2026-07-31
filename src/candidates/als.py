"""ALS matrix factorization (implicit feedback collaborative filtering).

Learns latent taste vectors for users and items from the full interaction
matrix, rather than only looking at direct co-purchase pairs like item-item
co-occurrence does — so it can generalize across items that never literally
co-occurred in a basket but are connected through shared user taste.

alpha=15 sets how much more we trust an observed purchase over an unobserved
one (confidence = 1 + alpha * count, per Hu et al. 2008) — a starting point
from the original paper's recommended range (15-40), not tuned here; real
tuning would sweep this against val Recall@K, which is future work, not core
to demonstrating the method.

filter_already_liked_items=False to stay consistent with the other two
methods: none of them exclude repurchases, and H&M's 12.9% repeat-purchase
rate means silently dropping them would make ALS's recall look artificially
worse in the comparison.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class ALSRecommender:
    def __init__(self, factors: int = 64, regularization: float = 0.01, alpha: float = 15.0, iterations: int = 15):
        self.model = AlternatingLeastSquares(
            factors=factors, regularization=regularization, alpha=alpha, iterations=iterations, random_state=42
        )

    def fit(self, train_df: pd.DataFrame) -> "ALSRecommender":
        n_users = train_df["user_idx"].max() + 1
        n_items = train_df["item_idx"].max() + 1
        counts = train_df.groupby(["user_idx", "item_idx"]).size().reset_index(name="count")
        self.user_items_ = sp.csr_matrix(
            (counts["count"].astype(np.float32), (counts["user_idx"], counts["item_idx"])),
            shape=(n_users, n_items),
        )
        self.model.fit(self.user_items_, show_progress=False)
        return self

    def recommend(self, user_idxs: list[int], k: int) -> dict[int, list[int]]:
        user_ids = np.array(user_idxs)
        ids, _scores = self.model.recommend(
            user_ids, self.user_items_[user_ids], N=k, filter_already_liked_items=False
        )
        return {u: ids[i].tolist() for i, u in enumerate(user_idxs)}
