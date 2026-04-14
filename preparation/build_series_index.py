"""
Run this once to build the FAISS index from output_with_descriptions.json
"""
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "series_index"
DATA_PATH = "output_with_descriptions.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight, fast, good quality

def build_text(series: dict) -> str:
    """
    Combine all fields into a single string for embedding.
    Richer text = better retrieval.
    """
    period_map = {"M": "Monthly", "Q": "Quarterly", "W": "Weekly", "D": "Daily", "A": "Annual"}
    period = period_map.get(series.get("PERIOD", ""), series.get("PERIOD", ""))

    return (
        f"{series['SERIES']}: {series['INDICATOR']}. "
        f"Category: {series['CATEGORY']} - {series['SUB-CATEGORY']}. "
        f"Unit: {series['UNITS']}. Frequency: {period}. "
        f"{series.get('description', '')}"
    )

def build_index():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        series_list = json.load(f)

    print(f"Loaded {len(series_list)} series from {DATA_PATH}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [build_text(s) for s in series_list]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # use IndexFlatIP (inner product) for cosine similarity after normalization
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # save index and metadata
    faiss.write_index(index, f"files/{INDEX_PATH}.faiss")
    with open(f"files/{INDEX_PATH}_meta.pkl", "wb") as f:
        pickle.dump({"series": series_list, "texts": texts}, f)

    print(f"Index built and saved:")
    print(f"  files/{INDEX_PATH}.faiss")
    print(f"  files/{INDEX_PATH}_meta.pkl")
    print(f"  Embedding dimension: {dimension}")
    print(f"  Total series indexed: {len(series_list)}")

if __name__ == "__main__":
    build_index()