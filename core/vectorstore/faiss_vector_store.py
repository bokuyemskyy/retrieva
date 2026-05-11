import faiss  # type: ignore
import numpy as np
import os
import pickle
from typing import List, Dict, Any

from core.vectorstore.base_vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_dim: int,
        index_path: str = "data/vectorstore/faiss.index",
    ):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        self.meta_path = index_path + ".meta.pkl"

        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")

        vectors = np.array(embeddings).astype("float32")

        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected dim {self.embedding_dim}, got {vectors.shape[1]}"
            )

        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_embedding: List[float], k: int = 5):
        query = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            results.append(
                {
                    "content": self.metadata[idx]["content"],
                    "metadata": self.metadata[idx]["metadata"],
                    "score": float(dist),
                }
            )

        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.metadata = pickle.load(f)
