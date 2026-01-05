from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.chunker import RagChunk
from rag.index import RagIndex


def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings


def build_index(chunks: List[RagChunk], model_name: str) -> RagIndex:
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, model_name)
    return RagIndex(embeddings=embeddings, chunks=chunks, model_name=model_name)


def retrieve(index: RagIndex, query: str, k: int = 5) -> List[Tuple[RagChunk, float]]:
    model = SentenceTransformer(index.model_name)
    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(index.embeddings, query_emb)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(index.chunks[i], float(scores[i])) for i in top_idx]
