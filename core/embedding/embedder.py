from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.models import Chunk, Embedding
import time


@dataclass
class EmbedderConfig:
    provider: str
    model_name: str

    api_key: str | None = None
    base_url: str | None = None


class BaseEmbedder(ABC):
    provider: str
    model_name: str
    _vector_size: int | None = None

    def __init__(self, config: EmbedderConfig):
        self.provider = config.provider
        self.model_name = config.model_name

    @abstractmethod
    def embed_query(self, text: str) -> Embedding:
        raise NotImplementedError

    @abstractmethod
    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding | None]:
        raise NotImplementedError

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            self._vector_size = len(self.embed_query("vector_size_probe"))
        return self._vector_size


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        import ollama

        self.client = ollama.Client(host=config.base_url)

    def embed_query(self, text: str) -> Embedding:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embed(model=self.model_name, input=[text])
                return response["embeddings"][0]
            except Exception as e:
                print(
                    f"Query embedding failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(1)

        print(f"Query embedding failed for text: {text[:50]}...")
        return [0.0] * self.vector_size

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding | None]:
        if not chunks:
            return []

        texts = [c.content for c in chunks]

        try:
            response = self.client.embed(input=texts, model=self.model_name)
            return response["embeddings"]
        except Exception as batch_error:
            print(
                f"Batch embedding failed for model {self.model_name} ({batch_error})."
            )
            print("Falling back to individual chunk processing.")

        safe_embeddings: list[Embedding | None] = []
        for text in texts:
            try:
                response = self.client.embed(input=[text], model=self.model_name)
                safe_embeddings.append(response["embeddings"][0])
            except Exception as e:
                print(f"\nCould not embed text: {text[:100]!r}")
                print(f"Reason: {e}\n")
                safe_embeddings.append(None)

        return safe_embeddings


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def embed_query(self, text: str) -> Embedding:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                res = self.client.embeddings.create(input=[text], model=self.model_name)
                return res.data[0].embedding
            except Exception as e:
                print(
                    f"Query embedding failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(1)

        print(f"Query embedding permanently failed for text: {text[:50]}...")
        return [0.0] * self.vector_size

    def embed_chunks(self, chunks: list[Chunk]) -> list[Embedding | None]:
        if not chunks:
            return []

        texts = [c.content for c in chunks]
        all_embeddings: list[Embedding | None] = []

        BATCH_SIZE = 50

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            success = False

            for attempt in range(3):
                try:
                    res = self.client.embeddings.create(
                        input=batch, model=self.model_name
                    )
                    all_embeddings.extend([data.embedding for data in res.data])
                    success = True
                    break
                except Exception as e:
                    print(
                        f"Batch embedding failed (batch {i // BATCH_SIZE}, attempt {attempt + 1}): {e}"
                    )
                    time.sleep(1)

            if not success:
                print(
                    f"Batch {i // BATCH_SIZE} failed. Falling back to individual processing for this batch..."
                )
                for text in batch:
                    try:
                        res = self.client.embeddings.create(
                            input=[text], model=self.model_name
                        )
                        all_embeddings.append(res.data[0].embedding)
                    except Exception as e:
                        print(f"Could not embed individual chunk: {e}")
                        all_embeddings.append(None)

        return all_embeddings


class EmbedderFactory:
    _registry: dict[str, type[BaseEmbedder]] = {}

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
