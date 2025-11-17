# src/indexer/chunker.py
from sentence_transformers import SentenceTransformer, util

def adaptive_semantic_chunking(paragraphs: list[str],
                               model_name: str,
                               min_len=300,
                               max_len=800,
                               sim_threshold=0.75) -> list[dict]:
    """
    Adaptive semantic chunking based on cosine similarity between paragraphs.
    Returns a list of {'id': str, 'text': str}
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    chunks, current_chunk = [], paragraphs[0]

    for i in range(1, len(paragraphs)):
        sim = float(util.cos_sim(embeddings[i-1], embeddings[i]))
        if len(current_chunk) < max_len and sim > sim_threshold:
            current_chunk += " " + paragraphs[i]
        else:
            if len(current_chunk) > min_len:
                chunks.append(current_chunk)
            current_chunk = paragraphs[i]
    chunks.append(current_chunk)

    return [{"id": f"chunk_{i+1}", "text": ch} for i, ch in enumerate(chunks)]
