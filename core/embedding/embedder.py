from abc import ABC, abstractmethod
from typing import List

from core.models import Chunk


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> None:
        pass


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        client,
        model: str = "text-embedding-3-small",
        vector_size: int = 1536,
    ) -> None:
        self._client = client
        self._model = model
        self.vector_size = vector_size

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ").strip()
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.content.replace("\n", " ").strip() for c in chunks]
        response = self._client.embeddings.create(input=texts, model=self._model)

        for chunk, item in zip(chunks, response.data):
            chunk.embedding = item.embedding


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

    def embed_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        texts = [c.content for c in chunks]
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=64)

        for chunk, vec in zip(chunks, vectors):
            chunk.embedding = vec.tolist()
