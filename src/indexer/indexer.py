import os
import yaml
import numpy as np
from pathlib import Path
from .extract import extract_text_from_pdf
from .preprocess import clean_text, normalize_dsa_terms
from .chunker import adaptive_semantic_chunking
from .embedder import Embedder
from .store_pinecone import PineconeStore
from dotenv import load_dotenv
import re
from typing import Dict, Any
from src.utils.config_loader import load_project_config


config: Dict[str, Any] = load_project_config("configs/config.yaml")
chunk_config = config.get("chunking", {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "min_length": 100,
    "max_length": 1000
})


def expand_env_vars(value):
    if isinstance(value, str):
        return re.sub(r"\$\{([^}]+)\}", lambda m: os.getenv(m.group(1), m.group(0)), value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value



class Indexer:
    def __init__(self, config_path="configs/config.yaml"):
        #  Load .env for Pinecone + other settings
        load_dotenv()

        #  Load YAML config and expand ${VAR}
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["indexer"]
        cfg = expand_env_vars(cfg)

        self.cfg = cfg
        self.embedder = Embedder(cfg["embeddings"]["model"], cfg["embeddings"]["cache_dir"])

   
    def build_index(self):
        pdf_root = Path(self.cfg["data_dirs"]["pdf_dir"])
        if not pdf_root.exists():
            raise FileNotFoundError(f"❌ PDF folder not found: {pdf_root.resolve()}")

        all_chunks, all_embeddings = [], []

        # 🔍 Recursively search for all PDFs in subfolders
        pdf_files = list(pdf_root.rglob("*.pdf"))
        if not pdf_files:
            print("⚠️ No PDF files found in the directory.")
            return

        print(f"📚 Found {len(pdf_files)} PDFs to index.\n")

        for pdf_path in pdf_files:
            rel_path = pdf_path.relative_to(pdf_root)
            print(f"📘 Indexing: {rel_path}")

            # 🧩 Step 1: Extract text
            pages = extract_text_from_pdf(str(pdf_path))
            text_list = [normalize_dsa_terms(clean_text(p["content"])) for p in pages]

            # 🧩 Step 2: Chunk text
            chunks = adaptive_semantic_chunking(
                text_list,
                self.cfg["embeddings"]["model"],
                min_len=self.cfg["chunking"]["min_chunk_size"], # type: ignore
                max_len=self.cfg["chunking"]["max_chunk_size"], # type: ignore
                sim_threshold=self.cfg["chunking"]["similarity_threshold"]
            )

            # 🧩 Step 3: Generate embeddings
            embeddings = self.embedder.embed_texts(chunks)
            for ch in chunks:
                ch["source_pdf"] = str(rel_path)
            all_chunks.extend(chunks)
            all_embeddings.append(embeddings)

        # step 4: Stack all embeddings
        all_embeddings = np.vstack(all_embeddings)

        # Step 5: Pinecone vector store
        store_cfg = self.cfg["vector_store"]
        store = PineconeStore(
            index_name=store_cfg["index_name"],
            dimension=store_cfg["dimension"],
            namespace=store_cfg.get("namespace", "main")
        )

        # Step 6: Upload to Pinecone
        store.add(all_embeddings, all_chunks)
        print("\n✅ Index build completed successfully.")
