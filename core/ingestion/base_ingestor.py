from abc import ABC, abstractmethod
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .models import Document, Chunk


class BaseIngestor(ABC):
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=150
        )

    @abstractmethod
    def ingest(self, document: Document) -> List[Chunk]:
        pass
