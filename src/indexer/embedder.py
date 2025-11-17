# src/indexer/embedder.py
from sentence_transformers import SentenceTransformer
import hashlib, os, json, numpy as np

def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

class Embedder:
    def __init__(self, model_name: str, cache_dir: str):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "embeddings_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=4)

    def embed_texts(self, chunks: list[dict]) -> np.ndarray:
        embeddings = []
        for ch in chunks:
            h = _hash_text(ch["text"])
            if h in self.cache:
                emb = np.array(self.cache[h])
            else:
                emb = self.model.encode(ch["text"], convert_to_numpy=True)
                self.cache[h] = emb.tolist()
            embeddings.append(emb)
        self._save_cache()
        return np.array(embeddings)
