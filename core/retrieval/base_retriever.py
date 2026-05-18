from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        pass
