from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional
import base64
import requests


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


class BaseVLM(ABC):
    @abstractmethod
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        raise NotImplementedError

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass


class NullVLM(BaseVLM):
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        return ""


class OllamaVLM(BaseVLM):
    def __init__(
        self,
        model_name: str = "llava",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url

    def load(self) -> None:
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "keep_alive": "5m"},
                timeout=5,
            )
        except requests.exceptions.RequestException:
            pass

    def unload(self) -> None:
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=5,
            )
        except requests.exceptions.RequestException:
            pass

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = (
            "Describe this image in detail. "
            "Include all visible text, objects, layout, colors, and any "
            "data or diagrams present. Be thorough and precise."
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()

        except requests.exceptions.RequestException as e:
            print(f"Ollama VLM request failed: {e}")
            return ""


class ImageCaptioner:
    def __init__(
        self,
        vlm: Optional[BaseVLM] = None,
        use_ocr: bool = True,
        ocr_lang: str = "eng",
    ) -> None:
        self._vlm = vlm or NullVLM()
        self._use_ocr = use_ocr
        self._ocr_lang = ocr_lang

    @property
    def vlm(self) -> BaseVLM:
        return self._vlm

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
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img, lang=self._ocr_lang).strip()
