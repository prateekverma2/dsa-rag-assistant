# src/indexer/preprocess.py
import re

def clean_text(text: str) -> str:
    """Basic cleaning: remove headers, multiple spaces, weird characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page\s*\d+', '', text)
    text = re.sub(r'[^a-zA-Z0-9.,;(){}[\]\s]', '', text)
    return text.strip()

def normalize_dsa_terms(text: str) -> str:
    """Normalize common DSA terms."""
    text = text.replace("O(", "Big O of ").replace(")", "")
    return text
