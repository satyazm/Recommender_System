"""Recently-purchased / repurchase candidates.

The simplest possible personalized signal: a user's own most-recently-bought
items, most-recent first. Trivial to compute, but per H&M's 12.9% repeat-
purchase rate (found during profiling) and confirmed by top Kaggle solutions
for this exact dataset, "will they buy it again" is one of the strongest
single signals available — worth including precisely because it's cheap, not
despite it.
"""
import pandas as pd


class RecencyRecommender:
    def fit(self, train_df: pd.DataFrame) -> "RecencyRecommender":
        ordered = train_df.sort_values("t_dat", ascending=False)
        self.user_history_ = ordered.groupby("user_idx")["item_idx"].apply(
            lambda items: list(dict.fromkeys(items))  # dedupe, keep most-recent-first order
        ).to_dict()
        return self

    def recommend(self, user_idxs: list[int], k: int) -> dict[int, list[int]]:
        return {u: self.user_history_.get(u, [])[:k] for u in user_idxs}
