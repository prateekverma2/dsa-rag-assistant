# src/indexer/store_pinecone.py
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

load_dotenv() 
api_key = os.getenv("PINECONE_API_KEY")
print(f"🔍 store_pinecone sees key prefix: {api_key[:8]!r}")
        
class PineconeStore:
    def __init__(self, index_name, dimension, namespace="default"):
        # ✅ Always load .env from project root
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

        api_key = os.getenv("PINECONE_API_KEY")
        region = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        if not api_key:
            raise RuntimeError("❌ No PINECONE_API_KEY found in .env")

        print(f"🔍 store_pinecone sees key prefix: '{api_key[:8]}'")

        self.index_name = index_name
        self.namespace = namespace

        
        self.pc = Pinecone(api_key=api_key)

      
        try:
            indexes_info = self.pc.list_indexes()
            existing_indexes = [i["name"] for i in indexes_info]
        except Exception as e:
            print("⚠️ Failed to list indexes (continuing anyway):", e)
            existing_indexes = []  # fallback empty list

        # ✅ Create index if missing
        if index_name not in existing_indexes:
            print(f"🆕 Creating Pinecone index: {index_name}")
            cloud = "aws" if "us" in region else "gcp"
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            print(f"✅ Index created successfully: {index_name}")
        else:
            print(f"✅ Using existing index: {index_name}")

        # ✅ Connect to index
        try:
            self.index = self.pc.Index(index_name)
            print(f"🎯 Connected to Pinecone index: {index_name}")
        except Exception as e:
            print("❌ Could not connect to index; check name and region.")
            raise e

    def add(self, embeddings: np.ndarray, metadata: list[dict], batch_size: int = 100):
        """
        Safely upload embeddings to Pinecone in small batches.
        Pinecone's request size limit is ~4MB, so we chunk uploads.
        """
        total = len(embeddings)
        print(f"📤 Uploading {total} vectors to Pinecone in batches of {batch_size}...")

        for start in tqdm(range(0, total, batch_size), desc="🔼 Upserting"):
            end = start + batch_size
            batch_vectors = [
                {"id": str(i), "values": emb.tolist(), "metadata": meta}
                for i, (emb, meta) in enumerate(zip(embeddings[start:end], metadata[start:end]), start)
            ]
            self.index.upsert(vectors=batch_vectors, namespace=self.namespace)

        print(f"✅ Uploaded {total} vectors to Pinecone namespace: {self.namespace}")

    def query(self, query_vector: np.ndarray, top_k: int = 5):
        res = self.index.query(
            vector=query_vector.tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace
        )
        return [
            {"text": match["metadata"].get("text", ""), "score": match["score"]}
            for match in res["matches"]
        ]
