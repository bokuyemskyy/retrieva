from __future__ import annotations

from pathlib import Path
from typing import List

import fitz

from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from models import Chunk, Document, Modality


def _overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    intersection = a & b
    if intersection.is_empty:
        return 0.0
    smaller = min(a.get_area(), b.get_area())
    return intersection.get_area() / smaller if smaller > 0 else 0.0


def _is_claimed(
    rect: fitz.Rect, claimed: List[fitz.Rect], threshold: float = 0.5
) -> bool:
    return any(_overlap_ratio(rect, c) > threshold for c in claimed)


class DocumentProcessor(BaseFileProcessor):
    def __init__(
        self,
        chunker: Chunker,
        image_captioner: ImageCaptioner,
        min_image_area: int = 25_000,
        overlap_threshold: float = 0.5,
    ) -> None:
        self.chunker = chunker
        self.image_captioner = image_captioner
        self._min_image_area = min_image_area
        self._overlap_threshold = overlap_threshold

    def ingest(self, document: Document) -> List[Chunk]:
        path = Path(document.source_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(document.source_path)

        doc = fitz.open(str(path))
        chunks: List[Chunk] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            base_meta = {"page": page_idx}
            page_area = page.rect.width * page.rect.height
            claimed: List[fitz.Rect] = []

            table_rects: List[fitz.Rect] = []

            table_finder = page.find_tables()

            if table_finder is not None:
                for tab_idx, table in enumerate(table_finder.tables):
                    md_table = table.to_markdown()
                    if not md_table:
                        continue
                    chunks.extend(
                        self.chunker.chunk(
                            content=f"Table from page {page_idx}:\n\n{md_table}",
                            document=document,
                            modality=Modality.PDF_TABLE,
                            metadata={**base_meta, "table_index": tab_idx},
                        )
                    )
                    table_rects.append(fitz.Rect(table.bbox))
                    claimed.append(fitz.Rect(table.bbox))

            text_parts: List[str] = []
            for block in page.get_text("blocks"):
                if block[6] != 0:
                    continue
                block_rect = fitz.Rect(block[:4])
                if _is_claimed(block_rect, table_rects, self._overlap_threshold):
                    continue
                text_parts.append(block[4])

            text = "\n\n".join(t.strip() for t in text_parts if t.strip())
            if text:
                chunks.extend(
                    self.chunker.chunk(
                        content=text,
                        document=document,
                        modality=Modality.PDF_TEXT,
                        metadata=base_meta,
                    )
                )

            for img_meta in page.get_images(full=True):
                xref = img_meta[0]

                img_rects = page.get_image_rects(xref)

                for img_rect in img_rects:
                    if _is_claimed(img_rect, claimed, self._overlap_threshold):
                        continue

                    new_chunks = self._process_image(
                        doc, xref, document, base_meta, page_area
                    )

                    if new_chunks:
                        chunks.extend(new_chunks)
                        claimed.append(img_rect)

            for cluster_idx, bbox in enumerate(
                page.cluster_drawings(x_tolerance=10, y_tolerance=10)
            ):
                w, h = bbox.width, bbox.height
                if w * h < self._min_image_area:
                    continue
                if _is_claimed(bbox, claimed, self._overlap_threshold):
                    continue

                pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(2, 2))
                vlm_text = self.image_captioner.process(pix.tobytes("png"), ext="png")
                if vlm_text:
                    chunks.extend(
                        self.chunker.chunk(
                            content=vlm_text,
                            document=document,
                            modality=Modality.PDF_IMAGE,
                            metadata={
                                **base_meta,
                                "cluster_index": cluster_idx,
                                "is_vector_plot": True,
                                "image_width": w,
                                "image_height": h,
                            },
                        )
                    )
                    claimed.append(bbox)

        doc.close()
        return chunks

    def _process_image(
        self,
        doc: fitz.Document,
        xref: int,
        document: Document,
        base_meta: dict,
        page_area: float,
    ) -> List[Chunk]:
        raw = doc.extract_image(xref)
        image_bytes: bytes = raw["image"]
        ext: str = raw.get("ext", "png")
        w, h = raw.get("width", 0), raw.get("height", 0)

        if w * h < self._min_image_area or (w * h) < (page_area * 0.02):
            return []

        text = self.image_captioner.process(image_bytes, ext=ext)
        if not text:
            return []

        return self.chunker.chunk(
            content=text,
            document=document,
            modality=Modality.PDF_IMAGE,
            metadata={
                **base_meta,
                "image_xref": xref,
                "image_ext": ext,
                "image_width": w,
                "image_height": h,
            },
        )
