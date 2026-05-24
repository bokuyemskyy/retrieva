from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import Chunk, Document


class BaseFileProcessor(ABC):
    @abstractmethod
    def ingest(self, document: Document) -> List[Chunk]:
        raise NotImplementedError
