from typing import List

from core.embedding.base_embedder import BaseEmbedder
from core.model.chunk import Chunk


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        vector_size: int = 1024,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required") from exc

        self._model = SentenceTransformer(model, device=device)
        self.vector_size = vector_size

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=64)

        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec.tolist()

        return chunks
