from abc import ABC, abstractmethod
from typing import List

from core.model.chunk import Chunk


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk], top_n: int) -> List[Chunk]:
        pass
