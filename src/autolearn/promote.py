import os
import json
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

class PromoteLearner:
    """
    Minimal persistence for approved Q&A (you can later re-index these to Pinecone in batches).
    """
    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("LEARN_PROMOTE_PATH", "learn_queue/approved.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def promote(self, question: str, answer: str, meta: Dict[str, Any] | None = None) -> None:
        entry = {
            "question": question,
            "answer": answer,
            "meta": meta or {},
            "status": "APPROVED",
            "ts": datetime.utcnow().isoformat() + "Z"
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
