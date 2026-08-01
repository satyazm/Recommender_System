"""Phase 6: cold-start artifacts.

Two independent gaps in what Phase 5 serves:
1. New USER fallback is currently pure global popularity — no personalization
   signal at all, even though we may know the user's age from customers.csv
   (true even for the 9,699 customers found in Phase 1 with zero transactions).
   Fix: age-bucketed popularity, a cheap demographic prior that's still much
   better than one global list for everyone.
2. New ITEM lookup currently 404s for any of the 995 articles with zero
   transactions (found in Phase 1) — they were never in the ALS-trained
   catalog, so they have no collaborative embedding. Content-based embeddings
   (TF-IDF + SVD over product attributes, same technique as Phase 2's
   ContentRecommender) don't have this problem: they only need the item's own
   attributes, not any purchase history. This builds that embedding for
   EVERY article — warm and cold — as the fallback index for /similar_items.
"""
import numpy as np
import pandas as pd
import faiss
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MODELS_DIR = "models"
DATA_DIR = "data/processed"

TEXT_COLUMNS = [
    "product_type_name", "product_group_name", "graphical_appearance_name",
    "colour_group_name", "perceived_colour_master_name", "department_name",
    "index_name", "section_name", "garment_group_name", "detail_desc",
]

# --- 1. age-bucketed popularity ---
txns = pd.read_parquet(f"{DATA_DIR}/transactions.parquet")
train = txns[txns["split"] == "train"]
customers = pd.read_csv("data/raw/hm/customers.csv", dtype={"customer_id": "category"})
customers[["customer_id", "age"]].to_parquet(f"{MODELS_DIR}/customer_age.parquet", index=False)

bins = [0, 25, 35, 45, 55, 200]
labels = ["under_25", "25_34", "35_44", "45_54", "55_plus"]
customers["age_bucket"] = pd.cut(customers["age"], bins=bins, labels=labels, right=False)

train_with_age = train.merge(customers[["customer_id", "age_bucket"]], on="customer_id", how="left")
age_pop = {}
for bucket in labels:
    top = train_with_age[train_with_age["age_bucket"] == bucket]["article_id"].value_counts().head(12)
    age_pop[bucket] = top.index.astype(str).tolist()
pd.Series(age_pop).to_json(f"{MODELS_DIR}/age_bucket_popularity.json")
print("Age-bucket popularity:")
for bucket, items in age_pop.items():
    print(f"  {bucket:<10} top item {items[0]}")

# --- 2. content embeddings for EVERY article, warm or cold ---
articles = pd.read_csv("data/raw/hm/articles.csv", dtype={"article_id": "int32"})
articles[TEXT_COLUMNS] = articles[TEXT_COLUMNS].fillna("")
text = articles[TEXT_COLUMNS].agg(" ".join, axis=1)

tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
tfidf_matrix = tfidf.fit_transform(text)
svd = TruncatedSVD(n_components=64, random_state=42)
embeddings = normalize(svd.fit_transform(tfidf_matrix).astype(np.float32))

np.save(f"{MODELS_DIR}/content_item_embeddings.npy", embeddings)
articles[["article_id"]].to_parquet(f"{MODELS_DIR}/all_article_ids.parquet", index=False)

content_index = faiss.IndexFlatIP(embeddings.shape[1])
content_index.add(embeddings)
faiss.write_index(content_index, f"{MODELS_DIR}/content_index.faiss")

n_train_items = pd.read_parquet(f"{DATA_DIR}/item_id_map.parquet").shape[0]
print(f"\nContent index: {content_index.ntotal:,} articles total "
      f"({content_index.ntotal - n_train_items:,} cold / never-transacted, covered here for the first time)")
