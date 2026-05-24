from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Any, Optional

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


class MoondreamVLM(BaseVLM):
    def __init__(self, model_id: str = "vikhyatk/moondream2") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model: Any = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16,
        )

        self._model.to("cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._torch = torch

    def load(self) -> None:
        target = "cuda" if self._torch.cuda.is_available() else "cpu"
        self._model.to(target)

    def unload(self) -> None:
        self._model.to("cpu")
        self._torch.cuda.empty_cache()

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        enc_image = self._model.encode_image(img)

        prompt = (
            "Describe this image in detail. "
            "Include all visible text, objects, layout, colors, and any "
            "data or diagrams present. Be thorough and precise."
        )
        answer = self._model.answer_question(enc_image, prompt, self._tokenizer)

        del enc_image
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        return answer.strip()


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
