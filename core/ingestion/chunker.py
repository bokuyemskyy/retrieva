from __future__ import annotations

from typing import List

from ..model.chunk import Chunk, Modality


class Chunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._step = chunk_size - chunk_overlap

    def chunk(
        self,
        text: str,
        source_path: str,
        modality: Modality,
        base_metadata: dict | None = None,
    ) -> List[Chunk]:
        text = text.strip()
        if not text:
            return []

        base_metadata = base_metadata or {}
        chunks: List[Chunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            window = text[start:end].strip()

            if window:
                chunks.append(
                    Chunk(
                        content=window,
                        source_path=source_path,
                        modality=modality,
                        metadata={
                            **base_metadata,
                            "chunk_index": chunk_index,
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )
                chunk_index += 1

            start += self._step

        return chunks
