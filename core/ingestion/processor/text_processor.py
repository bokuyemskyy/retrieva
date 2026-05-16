from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.model.chunk import Chunk, Modality


class TextProcessor(BaseFileProcessor):
    """
    Ingests plain-text files (.txt, .md, etc).
    """

    def __init__(
        self,
        chunker: Optional[Chunker] = None,
        image_processor: Optional[ImageCaptioner] = None,
    ) -> None:
        super().__init__(chunker=chunker, image_processor=image_processor)

    def ingest(self, file_path: str) -> List[Chunk]:
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Text file not found: {path}")

        text = path.read_text(encoding="utf-8", errors="ignore")

        return self.chunker.chunk(
            text=text,
            source_path=str(path),
            modality=Modality.TEXT,
            base_metadata={"filename": path.name},
        )
