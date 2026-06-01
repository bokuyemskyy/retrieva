from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from core.models import Chunk
import requests  # type: ignore


@dataclass
class EmbedderConfig:
    provider: str
    model_name: str

    api_key: Optional[str] = None
    base_url: Optional[str] = None


class BaseEmbedder(ABC):
    provider: str
    model_name: str
    _vector_size: Optional[int] = None

    def __init__(self, config: EmbedderConfig):
        self.provider = config.provider
        self.model_name = config.model_name

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Optional[List[float]]]:
        raise NotImplementedError

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            self._vector_size = len(self.embed_query("vector_size_probe"))
        return self._vector_size


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

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
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)

        from openai import OpenAI

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

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
    _registry: Dict[str, type[BaseEmbedder]] = {}

    @classmethod
    def register(cls, provider: str, embedder_class: type[BaseEmbedder]):
        cls._registry[provider] = embedder_class

    @classmethod
    def create(cls, config: EmbedderConfig, validate: bool = True) -> BaseEmbedder:
        if config.provider not in cls._registry:
            raise ValueError(f"Unknown provider {config.provider}")

        embedder_cls = cls._registry[config.provider]
        client = embedder_cls(config)

        if validate:
            try:
                client.vector_size
            except Exception as e:
                raise RuntimeError(
                    f"Model validation failed for {config.model_name}: {str(e)}"
                )

        return client


EmbedderFactory.register("ollama", OllamaEmbedder)
EmbedderFactory.register("openai", OpenAIEmbedder)
