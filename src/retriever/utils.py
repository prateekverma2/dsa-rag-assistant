# src/retriever/utils.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingHelper:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str):
        """Convert query to embedding"""
        return np.array(self.model.encode(query, convert_to_tensor=False))
