from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from core.models import Chunk
import requests  # type: ignore


class BaseEmbedder(ABC):
    provider: str
    model_name: str
    vector_size: int

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Optional[List[float]]]:
        raise NotImplementedError


class OllamaEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        vector_size: int,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.provider = "ollama"
        self.model_name = model_name
        self.vector_size = vector_size
        self.base_url = base_url

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_chunks(self, chunks: List[Chunk]) -> List[Optional[List[float]]]:
        embeddings: List[Optional[List[float]]] = []
        for chunk in chunks:
            try:
                embeddings.append(self.embed_query(chunk.content))
            except Exception as e:
                print(f"Failed to embed chunk {chunk.chunk_id}: {e}")
                embeddings.append(None)

        return embeddings


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, vector_size: int, api_key: str) -> None:
        import openai

        self.client = openai.Client(api_key=api_key)

        self.provider = "openai"
        self.model_name = model_name
        self.vector_size = vector_size

    def embed_query(self, text: str) -> List[float]:
        res = self.client.embeddings.create(input=[text], model=self.model_name)
        return res.data[0].embedding

    def embed_chunks(self, chunks: List[Chunk]) -> List[Optional[List[float]]]:
        if not chunks:
            return []

        texts = [c.content for c in chunks]

        try:
            res = self.client.embeddings.create(input=texts, model=self.model_name)
            return [data.embedding for data in res.data]
        except Exception as e:
            print(f"Batch embedding failed for model {self.model_name}: {e}")
            return [None] * len(chunks)


class EmbedderFactory:
    _registry = {}
    _cache = {}

    @classmethod
    def register(cls, provider: str, embedder_class: type[BaseEmbedder]):
        cls._registry[provider] = embedder_class

    @classmethod
    def create(
        cls, provider: str, model_name: str, vector_size: int, **kwargs
    ) -> BaseEmbedder:
        if provider not in cls._registry:
            raise ValueError(f"Unknown provider '{provider}'.")

        cache_key = f"{provider}_{model_name}"
        if cache_key not in cls._cache:
            embedder_cls = cls._registry[provider]
            cls._cache[cache_key] = embedder_cls(
                model_name=model_name, vector_size=vector_size, **kwargs
            )

        return cls._cache[cache_key]


EmbedderFactory.register("ollama", OllamaEmbedder)
EmbedderFactory.register("openai", OpenAIEmbedder)
