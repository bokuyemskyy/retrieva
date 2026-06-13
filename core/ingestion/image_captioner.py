from __future__ import annotations

from dataclasses import dataclass, field
import io
from abc import ABC, abstractmethod
import re
from PIL import Image
import base64


DEFAULT_PROMPT = (
    "Analyze this scientific figure. For each observation, state what it means, not just what it looks like. "
    "Cover in flowing prose: the core finding; all relationships between variables (trends, peaks, saturation, crossovers); "
    "quantitative anchors such as key values, ratios, and thresholds; physical or mechanistic implications — not just 'A is higher than B' but why that matters. "
    "Transcribe all axis labels, units, legend entries, and annotations verbatim. "
    "Be dense and factual. Every sentence must state a distinct scientific fact."
)


_BBOX_PATTERN = re.compile(
    r"^\s*\[\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\]\s*$"
)


def _is_degenerate(text: str) -> bool:
    return not text.strip() or bool(_BBOX_PATTERN.match(text.strip()))


@dataclass
class VLMConfig:
    provider: str
    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    prompt: str = field(default=DEFAULT_PROMPT)
    max_retries: int = 3


class BaseVLM(ABC):
    provider: str
    model_name: str
    prompt: str
    max_retries: int

    def __init__(self, config: VLMConfig):
        self.provider = config.provider
        self.model_name = config.model_name
        self.prompt = config.prompt
        self.max_retries = config.max_retries

    @abstractmethod
    def _describe_once(self, image_bytes: bytes, mime_type: str) -> str:
        raise NotImplementedError

    def describe(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        for attempt in range(self.max_retries):
            result = self._describe_once(image_bytes, mime_type)
            if not _is_degenerate(result):
                return result

        print(
            f"{self.model_name} returned only degenerate responses after {self.max_retries} attempts"
        )

        return ""


class NullVLM(BaseVLM):
    def _describe_once(self, image_bytes: bytes, mime_type: str) -> str:
        return ""


class OllamaVLM(BaseVLM):
    def __init__(self, config: VLMConfig) -> None:
        super().__init__(config)
        import ollama

        self.client = ollama.Client(host=config.base_url)

    def _describe_once(self, image_bytes: bytes, mime_type: str) -> str:
        response = self.client.generate(
            model=self.model_name, prompt=self.prompt, images=[image_bytes]
        )
        return response["response"]


class OpenAIVLM(BaseVLM):
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        from openai import OpenAI

        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def _describe_once(self, image_bytes: bytes, mime_type: str) -> str:
        b64_image = base64.b64encode(image_bytes).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": self.prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{b64_image}",
                    },
                ],
            }
        ]
        response = self.client.responses.create(model=self.model_name, input=messages)
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
    def __init__(self, vlm: BaseVLM) -> None:
        self._vlm = vlm

    @property
    def vlm(self) -> BaseVLM:
        return self._vlm

    def process(self, image_bytes: bytes, ext: str = "png") -> str:
        mime = _EXT_TO_MIME.get(ext.lower().lstrip("."), "image/png")
        return self._vlm.describe(image_bytes, mime_type=mime)


class VLMFactory:
    _registry: dict[str, type[BaseVLM]] = {}

    @classmethod
    def register(cls, provider: str, vlm_class: type[BaseVLM]):
        cls._registry[provider] = vlm_class

    @classmethod
    def create(cls, config: VLMConfig, validate: bool = True) -> BaseVLM:
        if config.provider not in cls._registry:
            raise ValueError(f"Unknown provider {config.provider}")

        client = cls._registry[config.provider](config)

        if validate:
            try:
                buf = io.BytesIO()
                Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
                client.describe(buf.getvalue(), mime_type="image/png")
            except Exception as e:
                raise RuntimeError(
                    f"Model validation failed for {config.model_name}: {str(e)}"
                )

        return client


VLMFactory.register("ollama", OllamaVLM)
VLMFactory.register("openai", OpenAIVLM)
VLMFactory.register("null", NullVLM)
