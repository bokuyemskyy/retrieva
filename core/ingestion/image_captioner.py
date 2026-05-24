from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Optional


class BaseVLM(ABC):
    @abstractmethod
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        raise NotImplementedError


class NullVLM(BaseVLM):
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        return ""


class MoondreamVLM(BaseVLM):
    """
    Native Python VLM with no external Ollama required.
    """

    def __init__(self, model_id: str = "vikhyatk/moondream2") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(
            f"Loading {model_id} into memory... This might take a minute on first run."
        )

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        return answer.strip()


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
            print(ocr_text)

        vlm_text = self._vlm.describe(image_bytes, mime_type=mime)
        if vlm_text:
            parts.append(f"[VISUAL DESCRIPTION]\n{vlm_text}")
            print(vlm_text)

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
