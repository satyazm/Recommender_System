"""Item-item co-occurrence ("customers who bought X also bought Y").

H&M has no explicit order/session id, so (customer_id, day) is used as a basket
proxy — items bought by the same customer on the same day are a reasonable stand-in
for "bought together." Baskets larger than 20 items (0.6% of them, max 570) are
dropped: a handful of bulk-buy outliers would otherwise dominate the co-purchase
counts for items that happen to appear in them.

Co-occurrence counts are normalized per source item into P(j bought | i bought) —
an asymmetric conditional probability, not a symmetric similarity. This matters:
without normalization, popular items would co-occur with everything simply by
being popular, and the "co-occurrence" signal would collapse into the popularity
baseline it's supposed to improve on.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


class ItemItemRecommender:
    MAX_BASKET_SIZE = 20

    def fit(self, train_df: pd.DataFrame) -> "ItemItemRecommender":
        basket_id = train_df.groupby(["customer_id", "t_dat"], observed=True).ngroup()
        basket_sizes = basket_id.value_counts()
        valid_baskets = basket_sizes[basket_sizes <= self.MAX_BASKET_SIZE].index
        mask = basket_id.isin(valid_baskets)

        rows = basket_id[mask].values
        cols = train_df.loc[mask, "item_idx"].values
        n_items = train_df["item_idx"].max() + 1
        n_baskets = rows.max() + 1

        basket_item = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_baskets, n_items)
        )
        basket_item.data[:] = 1.0  # a basket buying item i twice still counts once

        co_occurrence = (basket_item.T @ basket_item).tocsr()
        item_basket_count = np.asarray(basket_item.sum(axis=0)).ravel()
        item_basket_count[item_basket_count == 0] = 1  # avoid div-by-zero for unreachable items

        # normalize each row i by how often item i appears in a basket at all -> P(j | i)
        inv_counts = sp.diags(1.0 / item_basket_count)
        self.affinity_ = (inv_counts @ co_occurrence).tocsr()
        self.affinity_.setdiag(0)
        self.affinity_.eliminate_zeros()

        self.user_history_ = train_df.groupby("user_idx")["item_idx"].apply(list).to_dict()
        return self

    def recommend(self, user_idxs: list[int], k: int) -> dict[int, list[int]]:
        recs = {}
        for u in user_idxs:
            history = self.user_history_.get(u, [])
            if not history:
                recs[u] = []
                continue
            scores = np.asarray(self.affinity_[history, :].sum(axis=0)).ravel()
            top_k = np.argpartition(scores, -k)[-k:]
            top_k = top_k[np.argsort(-scores[top_k])]
            recs[u] = top_k.tolist()
        return recs
