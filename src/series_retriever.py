import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

INDEX_PATH = "series_index"
MODEL_NAME = "all-MiniLM-L6-v2"

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SeriesRetriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(os.path.join(_BASE_DIR, f"../files/{INDEX_PATH}.faiss"))

        with open(os.path.join(_BASE_DIR, f"../files/{INDEX_PATH}_meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        self.series_list = meta["series"]
        print(f"SeriesRetriever loaded: {len(self.series_list)} series available")

    def get_all_series_ids(self):
        """
        Get all series ids.
        """
        return set([s['SERIES'] for s in self.series_list])

    def retrieve(self, query: str, top_k: int = 8) -> list[dict]:
        """
        Retrieve the most relevant series for a given query.

        Returns:
            list of dicts with series info + similarity score
        """
        query_vec = self.model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # if score < 0.25:  # skip those with similarity lower than 0.25
            #     continue
            series = self.series_list[idx].copy()
            series["similarity"] = float(score)
            results.append(series)

        return results

    def build_prompt_section(self, query: str, top_k: int = 8) -> str:
        """
        Generate a prompt section with only the relevant series.
        Replaces the full INDICATOR_GUIDE in the system prompt.
        """
        relevant = self.retrieve(query, top_k)

        period_map = {"M": "Monthly", "Q": "Quarterly", "W": "Weekly", "D": "Daily", "A": "Annual"}

        lines = [
            "Relevant FRED Economic Indicators for this query:",
            "Format: - SERIES ID - Indicator Name - Unit - Frequency",
        ]
        for s in relevant:
            period = period_map.get(s.get("PERIOD", ""), s.get("PERIOD", ""))
            lines.append(f"- {s['SERIES']} - {s['INDICATOR']} - {s['UNITS']} - {period}")

        return "\n".join(lines)

    def test_retrieval(self, query: str, top_k: int = 8):
        """
        Print retrieval results for a query. Useful for debugging.
        """
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = self.retrieve(query, top_k)
        for i, s in enumerate(results):
            print(f"{i+1}. [{s['similarity']:.3f}] {s['SERIES']}: {s['INDICATOR']}")


if __name__ == "__main__":
    retriever = SeriesRetriever()

    # test some queries
    test_queries = [
        "How is the economy doing?",
        "What's happening with inflation and prices?",
        "How's the real estate market?",
        "What's the trade balance with China?",
        "How is consumer confidence?",
        "What are interest rates doing?",
        "How's the job market?",
        "What's the dollar exchange rate?",
    ]

    for query in test_queries:
        retriever.test_retrieval(query, top_k=5)