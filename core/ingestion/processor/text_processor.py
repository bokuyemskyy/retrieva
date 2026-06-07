from __future__ import annotations

from pathlib import Path
from typing import List

from core.ingestion.chunker import BaseChunker

from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.models import Chunk, Modality, Document


class TextProcessor(BaseFileProcessor):
    def __init__(self, chunker: BaseChunker) -> None:
        self.chunker = chunker

    def ingest(self, document: Document) -> List[Chunk]:
        path = Path(document.source_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        text = path.read_text(encoding="utf-8", errors="ignore")

        return self.chunker.chunk(
            content=text,
            document=document,
            modality=Modality.TEXT,
        )
