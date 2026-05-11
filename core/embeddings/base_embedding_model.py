from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
