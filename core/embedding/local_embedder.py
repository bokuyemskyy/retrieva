from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from core.embeddings.base_embedding_model import BaseEmbeddingModel


class LocalEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.astype("float32")

    def embed_queries(self, texts: List[str]) -> List[List[float]]:

        return self._encode(texts).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        return self._encode(texts).tolist()
