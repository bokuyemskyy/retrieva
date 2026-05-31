from __future__ import annotations

from dataclasses import dataclass
import io
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from PIL import Image

import base64


@dataclass
class VLMConfig:
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class BaseVLM(ABC):
    provider: str
    model_name: str

    def __init__(self, config: VLMConfig):
        self.provider = config.provider
        self.model_name = config.model_name

    @abstractmethod
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        raise NotImplementedError


class NullVLM(BaseVLM):
    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        return ""


class OllamaVLM(BaseVLM):
    def __init__(self, config: VLMConfig) -> None:
        super().__init__(config)

        import ollama

        self.client = ollama.Client(host=config.base_url)

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        prompt = (
            "Describe this image in detail. "
            "Include all visible text, objects, layout, colors, and any "
            "data or diagrams present. Be thorough and precise."
        )
        
        response = self.client.generate(model=self.model_name, prompt=prompt, images=[image_bytes])
        return response["response"]


class OpenAIVLM(BaseVLM):
    def __init__(self, config: VLMConfig):
        super().__init__(config)

        from openai import OpenAI

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def describe(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
    ) -> str:
        b64_image = base64.b64encode(image_bytes).decode()

        messages: list[Any] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Describe this image in detail. "
                            "Include all visible text, objects, "
                            "layout, colors, and diagrams."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    },
                ],
            }
        ]

        response = self.client.responses.create(
            model=self.model_name,
            input=messages,
        )

        return response.output_text.strip()


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
    def __init__(
        self,
        vlm: BaseVLM,
        use_ocr: bool = True,
        ocr_lang: str = "eng",
    ) -> None:
        self._vlm = vlm
        self._use_ocr = use_ocr
        self._ocr_lang = ocr_lang

    @property
    def vlm(self) -> BaseVLM:
        return self._vlm

    def process(self, image_bytes: bytes, ext: str = "png") -> str:
        mime = _EXT_TO_MIME.get(ext.lower().lstrip("."), "image/png")
        parts: list[str] = []

        vlm_text = self._vlm.describe(image_bytes, mime_type=mime)
        if vlm_text:
            parts.append(f"Image description:\n{vlm_text}")

        if self._use_ocr:
            ocr_text = self._run_ocr(image_bytes)
            if ocr_text:
                parts.append(f"Image ocr:\n{ocr_text}")

        return "\n\n".join(parts)

    def _run_ocr(self, image_bytes: bytes) -> str:
        import pytesseract

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
        except Exception:
            return ""
        return pytesseract.image_to_string(img, lang=self._ocr_lang).strip()


class VLMFactory:
    _registry: Dict[str, type[BaseVLM]] = {}

    @classmethod
    def register(cls, provider: str, vlm_class: type[BaseVLM]):
        cls._registry[provider] = vlm_class

    @classmethod
    def create(cls, config: VLMConfig, validate: bool = True) -> BaseVLM:
        if config.provider not in cls._registry:
            raise ValueError(f"Unknown provider {config.provider}")

        vlm_class = cls._registry[config.provider]
        client = vlm_class(config)

        if validate:
            try:
                test_img = Image.new("RGB", (10, 10), "white")
                buf = io.BytesIO()
                test_img.save(buf, format="PNG")

                client.describe(buf.getvalue(), mime_type="image/png")
            except Exception as e:
                raise RuntimeError(
                    f"Model validation failed for {config.model_name}: {str(e)}"
                )

        return client


VLMFactory.register("ollama", OllamaVLM)
VLMFactory.register("openai", OpenAIVLM)
VLMFactory.register("null", NullVLM)
