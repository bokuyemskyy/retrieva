from typing import List
from pathlib import Path

from core.ingestion.base_ingestor import BaseIngestor, Document, DocumentChunk


class TextIngestor(BaseIngestor):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, source: str) -> List[Document]:
        path = Path(source)

        documents = []

        files = [path] if path.is_file() else list(path.glob("*.txt"))

        for file in files:
            text = file.read_text(encoding="utf-8", errors="ignore")

            if text.strip():
                documents.append(Document(content=text, metadata={"source": str(file)}))

        return documents

    def chunk(self, documents: List[Document]) -> List[DocumentChunk]:
        chunks = []

        for doc in documents:
            text = doc.content
            start = 0

            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        DocumentChunk(
                            content=chunk_text,
                            metadata={
                                **doc.metadata,
                                "start": start,
                                "end": end,
                            },
                        )
                    )

                start += self.chunk_size - self.chunk_overlap

        return chunks
