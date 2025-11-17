# base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Abstract base retriever."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks."""
        pass
