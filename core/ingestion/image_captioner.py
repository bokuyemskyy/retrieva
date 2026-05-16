from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional


class BaseVLM(ABC):
    @abstractmethod
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        pass


class NullVLM(BaseVLM):
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        return ""


class OllamaVLM(BaseVLM):
    """
    Requirements:
    ollama serve
    ollama pull llava
    """

    def __init__(
        self, model: str = "llava", host: str = "http://localhost:11434"
    ) -> None:
        import ollama

        self._client = ollama.Client(host=host)
        self._model = model

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        response = self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Describe this image in detail. "
                        "Include all visible text, objects, layout, colors, and any "
                        "data or diagrams present. Be thorough and precise."
                    ),
                    "images": [image_bytes],
                }
            ],
        )
        return response["message"]["content"].strip()


_EXT_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "tif": "image/tiff",
}


class ImageCaptioner:
    """
    Converts raw image bytes to plain text by running OCR or/and a VLM.

    Parameters
    vlm:
        A BaseVLM instance.
    use_ocr:
        Whether to run pytesseract OCR.
    ocr_lang:
        Tesseract language string.
    """

    def __init__(
        self,
        vlm: Optional[BaseVLM] = None,
        use_ocr: bool = True,
        ocr_lang: str = "eng",
    ) -> None:
        self._vlm = vlm or NullVLM()
        self._use_ocr = use_ocr
        self._ocr_lang = ocr_lang

    def process(self, image_bytes: bytes, ext: str = "png") -> str:
        mime = _EXT_TO_MIME.get(ext.lower().lstrip("."), "image/png")
        parts: list[str] = []

        if self._use_ocr:
            ocr_text = self._run_ocr(image_bytes)
            if ocr_text:
                parts.append(f"[OCR TEXT]\n{ocr_text}")

        vlm_text = self._vlm.describe(image_bytes, mime_type=mime)
        if vlm_text:
            parts.append(f"[VISUAL DESCRIPTION]\n{vlm_text}")

        return "\n\n".join(parts)

    def _run_ocr(self, image_bytes: bytes) -> str:
        try:
            import pytesseract  # type: ignore
            from PIL import Image
        except ImportError as exc:
            raise ImportError("OCR requires pytesseract and Pillow") from exc

        img = Image.open(io.BytesIO(image_bytes))
        text: str = pytesseract.image_to_string(img, lang=self._ocr_lang)
        return text.strip()
