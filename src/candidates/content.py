"""Content-based candidate generation via item embeddings.

Unlike ALS/co-occurrence, this method never needs to observe an interaction to
place an item in embedding space — its embedding comes purely from product
attributes (type, department, colour, garment group, description text). That's
exactly the property Phase 6 cold-start handling needs for brand-new articles,
which is why this method exists here even though it's the weakest of the three
on Recall@K for WARM users below: it's not competing to win this comparison,
it's the only one of the three that still works when collaborative signal is
completely absent.

TF-IDF + TruncatedSVD (not a learned neural embedding) is a deliberate
simplicity choice: it's fast, deterministic, and needs no training
infrastructure, at the cost of not capturing similarity beyond word overlap in
attribute text (e.g. it won't know "biker jacket" and "moto jacket" are
related unless the vocabulary happens to overlap).
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TEXT_COLUMNS = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_master_name", "department_name",
    "index_name", "section_name", "garment_group_name", "detail_desc",
]


class ContentRecommender:
    def __init__(self, articles_path: str, item_map: pd.DataFrame, n_components: int = 64, batch_size: int = 500):
        self.batch_size = batch_size
        articles = pd.read_csv(articles_path, dtype={"article_id": "int32"})
        articles[TEXT_COLUMNS] = articles[TEXT_COLUMNS].fillna("")
        text = articles[TEXT_COLUMNS].agg(" ".join, axis=1)

        tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
        tfidf_matrix = tfidf.fit_transform(text)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        full_embeddings = svd.fit_transform(tfidf_matrix)

        article_row = {aid: i for i, aid in enumerate(articles["article_id"])}
        # item_map is ordered by item_idx (0..n_items_train-1) -> pull embeddings in that order
        train_rows = [article_row[aid] for aid in item_map.sort_values("item_idx")["article_id"]]
        self.item_embeddings_ = normalize(full_embeddings[train_rows].astype(np.float32))

    def fit(self, train_df: pd.DataFrame) -> "ContentRecommender":
        user_history = train_df.groupby("user_idx")["item_idx"].apply(list)
        profiles = np.zeros((user_history.index.max() + 1, self.item_embeddings_.shape[1]), dtype=np.float32)
        for user_idx, items in user_history.items():
            profiles[user_idx] = self.item_embeddings_[items].mean(axis=0)
        self.user_profiles_ = normalize(profiles)
        return self

    def recommend(self, user_idxs: list[int], k: int) -> dict[int, list[int]]:
        recs = {}
        for start in range(0, len(user_idxs), self.batch_size):
            batch = user_idxs[start:start + self.batch_size]
            sims = self.user_profiles_[batch] @ self.item_embeddings_.T  # (batch, n_items)
            for row, u in enumerate(batch):
                scores = sims[row]
                top_k = np.argpartition(scores, -k)[-k:]
                top_k = top_k[np.argsort(-scores[top_k])]
                recs[u] = top_k.tolist()
        return recs
