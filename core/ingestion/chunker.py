from __future__ import annotations

from typing import List
from uuid import uuid4


from core.models import Chunk, Modality, Document


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._step = chunk_size - chunk_overlap

    def chunk(
        self,
        content: str,
        document: Document,
        modality: Modality,
        metadata: dict | None = None,
    ) -> List[Chunk]:
        content = content.strip()
        if not content:
            return []

        metadata = metadata or {}

        chunks: List[Chunk] = []

        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))

            raw_window = content[start:end]

            if end < len(content):
                last_space = raw_window.rfind(" ")
                if last_space > 0:
                    end = start + last_space
                    raw_window = content[start:end]

            window = raw_window.strip()

            if window:
                chunk_id = uuid4()

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        content=window,
                        document_id=document.document_id,
                        source_path=document.source_path,
                        workspace=document.workspace,
                        modality=modality,
                        metadata={
                            **metadata,
                            "chunk_index": chunk_index,
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )
                chunk_index += 1

            start += self._step

        return chunks
