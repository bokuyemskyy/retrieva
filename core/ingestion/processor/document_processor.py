from __future__ import annotations

from pathlib import Path
from typing import List

from core.ingestion.chunker import Chunker
from core.ingestion.image_captioner import ImageCaptioner
from core.ingestion.processor.base_file_processor import BaseFileProcessor
from models import Chunk, Document, Modality


class DocumentProcessor(BaseFileProcessor):
    def __init__(
        self,
        chunker: Chunker,
        image_captioner: ImageCaptioner,
        min_image_area: int = 25000,
    ) -> None:
        self.chunker = chunker
        self.image_processor = image_captioner
        self._min_image_area = min_image_area

    def ingest(self, document: Document) -> List[Chunk]:
        import fitz  # type: ignore

        path = Path(document.source_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(document.source_path)

        doc = fitz.open(str(path))
        chunks: List[Chunk] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            base_meta = {"page": page_idx}

            page_area = page.rect.width * page.rect.height

            text = page.get_text()
            text = text.strip() if isinstance(text, str) else ""
            if text:
                chunks.extend(
                    self.chunker.chunk(
                        content=text,
                        document=document,
                        modality=Modality.PDF_TEXT,
                        metadata=base_meta,
                    )
                )

            tables = page.find_tables()
            for tab_idx, table in enumerate(tables):
                md_table = table.to_markdown()
                if md_table:
                    chunks.extend(
                        self.chunker.chunk(
                            content=f"Table extracted from page {page_idx}:\n\n{md_table}",
                            document=document,
                            modality=Modality.PDF_TABLE,
                            metadata={
                                **base_meta,
                                "table_index": tab_idx,
                            },
                        )
                    )

            print("done1")
            for img_meta in page.get_images(full=True):
                xref = img_meta[0]
                chunks.extend(
                    self._process_image(doc, xref, document, base_meta, page_area)
                )
            print("full done")

            clusters = page.cluster_drawings(x_tolerance=10, y_tolerance=10)

            for cluster_idx, bbox in enumerate(clusters):
                w, h = bbox.width, bbox.height
                if w * h < self._min_image_area:
                    continue

                pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")

                doc_name = path.stem
                debug_file_name = f"{doc_name}_page_{page_idx}_plot_{cluster_idx}.png"
                debug_path = Path(".") / debug_file_name
                with open(debug_path, "wb") as f:
                    f.write(image_bytes)
                print(f"[DEBUG] Extracted plot saved to: {debug_path} ({w}x{h})")

                vlm_text = self.image_processor.process(image_bytes, ext="png")
                if vlm_text:
                    chunks.extend(
                        self.chunker.chunk(
                            content=vlm_text,
                            document=document,
                            modality=Modality.PDF_IMAGE,
                            metadata={
                                **base_meta,
                                "is_vector_plot": True,
                                "image_ext": "png",
                                "image_width": w,
                                "image_height": h,
                            },
                        )
                    )

        doc.close()
        return chunks

    def _process_image(
        self,
        doc,
        xref: int,
        document: Document,
        base_meta: dict,
        page_area: float,
    ) -> List[Chunk]:
        raw = doc.extract_image(xref)
        image_bytes: bytes = raw["image"]
        ext: str = raw.get("ext", "png")
        w, h = raw.get("width", 0), raw.get("height", 0)

        if w * h < self._min_image_area:
            return []

        if (w * h) < (page_area * 0.02):
            return []

        page_num = base_meta.get("page", 0)
        doc_name = Path(document.source_path).stem
        debug_file_name = f"{doc_name}_page_{page_num}_xref_{xref}.{ext}"
        debug_path = Path(".") / debug_file_name

        with open(debug_path, "wb") as f:
            f.write(image_bytes)
        print(f"[DEBUG] Extracted image saved to: {debug_path} ({w}x{h})")

        if w * h < self._min_image_area:
            return []

        text = self.image_processor.process(image_bytes, ext=ext)
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
