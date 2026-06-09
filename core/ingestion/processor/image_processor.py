from __future__ import annotations

from pathlib import Path

from core.ingestion.chunker import BaseChunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.models import Chunk, Modality, Document


class ImageProcessor(BaseFileProcessor):
    def __init__(
        self,
        chunker: BaseChunker,
        image_captioner: ImageCaptioner,
    ) -> None:
        self.image_captioner = image_captioner
        self.chunker = chunker

    def ingest(self, document: Document) -> list[Chunk]:
        path = Path(document.source_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(document.source_path)

        ext = path.suffix.lower()
        image_bytes = path.read_bytes()
        text = self.image_captioner.process(image_bytes, ext=ext.lstrip("."))

        return self.chunker.chunk(
            content=text,
            document=document,
            modality=Modality.IMAGE,
        )
