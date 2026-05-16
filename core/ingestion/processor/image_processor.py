from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.model.chunk import Chunk, Modality

_SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
}


class ImageProcessor(BaseFileProcessor):
    def __init__(
        self,
        chunker: Optional[Chunker] = None,
        image_processor: Optional[ImageCaptioner] = None,
    ) -> None:
        super().__init__(chunker=chunker, image_processor=image_processor)

    def ingest(self, file_path: str) -> List[Chunk]:
        path = Path(file_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension {ext}.")

        image_bytes = path.read_bytes()
        text = self.image_processor.process(image_bytes, ext=ext.lstrip("."))

        return self.chunker.chunk(
            text=text,
            source_path=str(path),
            modality=Modality.IMAGE,
            base_metadata={"filename": path.name, "image_ext": ext.lstrip(".")},
        )
