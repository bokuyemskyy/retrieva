from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from core.model.chunk import Chunk, Modality

_MIN_IMAGE_AREA = 4_000


class DocumentProcessor(BaseFileProcessor):
    def __init__(
        self,
        min_image_area: int = _MIN_IMAGE_AREA,
        chunker: Optional[Chunker] = None,
        image_processor: Optional[ImageCaptioner] = None,
    ) -> None:
        super().__init__(chunker=chunker, image_processor=image_processor)
        self._min_image_area = min_image_area

    def ingest(self, file_path: str) -> List[Chunk]:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise ImportError("PDFIngestor requires pymupdf.") from exc

        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc = fitz.open(str(path))
        chunks: List[Chunk] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            base_meta = {"page": page_idx, "filename": path.name}

            text = page.get_text().strip()
            if text:
                chunks.extend(
                    self.chunker.chunk(
                        text=text,
                        source_path=str(path),
                        modality=Modality.PDF_TEXT,
                        base_metadata=base_meta,
                    )
                )

            for img_meta in page.get_images(full=True):
                xref = img_meta[0]
                chunks.extend(
                    self._process_image(doc, xref, str(path), page_idx, base_meta)
                )

        doc.close()
        return chunks

    def _process_image(
        self,
        doc,
        xref: int,
        source_path: str,
        page_idx: int,
        base_meta: dict,
    ) -> List[Chunk]:
        raw = doc.extract_image(xref)
        image_bytes: bytes = raw["image"]
        ext: str = raw.get("ext", "png")

        w, h = raw.get("width", 0), raw.get("height", 0)
        if w * h < self._min_image_area:
            return []

        text = self.image_processor.process(image_bytes, ext=ext)
        if not text:
            return []

        return self.chunker.chunk(
            text=text,
            source_path=source_path,
            modality=Modality.PDF_IMAGE,
            base_metadata={
                **base_meta,
                "image_xref": xref,
                "image_ext": ext,
                "image_width": w,
                "image_height": h,
            },
        )
