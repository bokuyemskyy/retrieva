from abc import ABC, abstractmethod
from sentence_transformers import CrossEncoder
from core.models import Chunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        pass


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []

        pairs = [(query, c.content) for c in chunks]

        scores = self.model.predict(pairs)

        return [float(score) for score in scores]
