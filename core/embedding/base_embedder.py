from abc import ABC, abstractmethod
from typing import Dict, List

from core.model.chunk import Chunk


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> Dict[Chunk, List[float]]:
        pass
