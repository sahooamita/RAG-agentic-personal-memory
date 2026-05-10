"""SentenceTransformer embedding wrapper."""
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) numpy array of embeddings."""
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
