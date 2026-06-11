from __future__ import annotations

from pathlib import Path
import fitz  # type: ignore

import pymupdf4llm  # type: ignore

from core.ingestion.chunker import BaseChunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from models import Chunk, Document, Modality  # type: ignore


class DocumentProcessor(BaseFileProcessor):
    def __init__(
        self,
        chunker: BaseChunker,
        image_captioner: ImageCaptioner,
        min_image_area: int = 25_000,
    ) -> None:
        self.chunker = chunker
        self.image_captioner = image_captioner
        self._min_image_area = min_image_area

    def ingest(self, document: Document) -> list[Chunk]:
        path = Path(document.source_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(document.source_path)

        doc = fitz.open(str(path))
        chunks: list[Chunk] = []

        print(f"Ingesting: {document.source_path}")

        page_data = pymupdf4llm.to_markdown(
            doc,
            page_chunks=True,
            header=False,
            footer=False,
        )

        for page_idx, page_info in enumerate(page_data):
            print(f"Processing page {page_idx}/{len(doc)}")
            base_meta = {"page": page_idx}

            md_text = page_info["text"]
            if md_text.strip():
                chunks.extend(
                    self.chunker.chunk(
                        content=md_text,
                        document=document,
                        modality=Modality.PDF_TEXT,
                        metadata=base_meta,
                    )
                )

            page = doc[page_idx]
            for box in page_info.get("page_boxes", []):
                if box["class"] == "picture":
                    rect = fitz.Rect(box["bbox"])

                    if rect.get_area() >= self._min_image_area:
                        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
                        img_bytes = pix.tobytes("png")

                        vlm_caption = self.image_captioner.process(img_bytes, ext="png")

                        if vlm_caption:
                            chunks.extend(
                                self.chunker.chunk(
                                    content=f"Image Caption from page {page_idx}:\n{vlm_caption}",
                                    document=document,
                                    modality=Modality.PDF_IMAGE,
                                    metadata={
                                        **base_meta,
                                        "image_width": rect.width,
                                        "image_height": rect.height,
                                    },
                                )
                            )

        doc.close()
        return chunks
