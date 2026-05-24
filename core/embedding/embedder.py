from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List

from models import Chunk


class BaseEmbedder(ABC):
    vector_size: int

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> None:
        raise NotImplementedError

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        vector_size: int = 1024,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device="cpu")
        self._torch = torch
        self.vector_size = vector_size

    def load(self) -> None:
        target = "cuda" if self._torch.cuda.is_available() else "cpu"
        self._model = self._model.to(target)

    def unload(self) -> None:
        self._model = self._model.to("cpu")
        self._torch.cuda.empty_cache()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.content for c in chunks]
        total_chars = sum(len(t) for t in texts)
        print(f"Embedding {len(chunks)} chunks ({total_chars} chars)…")

        start = time.time()
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )
        print(f"Finished embedding in {time.time() - start:.2f}s.")

        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec.tolist()
