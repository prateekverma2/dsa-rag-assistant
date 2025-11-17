import os
import json
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

class LearnQueue:
    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("LEARN_QUEUE_PATH", "learn_queue/pending.jsonl")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def add(self, question: str, answer: str, reason: str = "") -> None:
        entry: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "reason": reason,
            "ts": datetime.utcnow().isoformat() + "Z"
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
