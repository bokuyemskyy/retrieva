from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import Chunk, Document
from core.ingestion.chunker import Chunker


class BaseFileProcessor(ABC):
    @abstractmethod
    def ingest(self, document: Document, chunker: Chunker) -> List[Chunk]:
        raise NotImplementedError
