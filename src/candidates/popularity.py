"""Popularity baseline — the floor every other candidate-gen method must beat.

Not personalized at all: every user gets the same top-N most-purchased items
from train. Trivial to compute, always available (never cold), and the
standard reference point recsys papers report lift against.
"""
import pandas as pd


class PopularityRecommender:
    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        self.ranked_items_ = train_df["item_idx"].value_counts().index.tolist()
        return self

    def recommend(self, user_idxs: list[int], k: int) -> dict[int, list[int]]:
        top_k = self.ranked_items_[:k]
        return {u: top_k for u in user_idxs}
