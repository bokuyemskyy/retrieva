from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
