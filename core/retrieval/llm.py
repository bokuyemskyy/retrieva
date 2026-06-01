from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMConfig:
    provider: str

    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    temperature: float = 0.0
    max_tokens: Optional[int] = 1024
    top_p: float = 1.0


class BaseLLM(ABC):
    provider: str
    config: LLMConfig

    def __init__(self, config: LLMConfig):
        self.provider = config.provider
        self.config = config

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_stream(self, system_prompt: str, user_prompt: str):
        raise NotImplementedError


class StandardLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)

        from openai import OpenAI

        base_url = config.base_url
        api_key = config.api_key

        if config.provider == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            api_key = api_key or "ollama"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def _get_shared_args(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        return kwargs

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = self._get_shared_args(system_prompt, user_prompt)
        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            return (
                f"LLM request failed for {self.provider}/{self.config.model_name}: {e}"
            )

    def generate_stream(self, system_prompt: str, user_prompt: str):
        kwargs = self._get_shared_args(system_prompt, user_prompt)
        kwargs["stream"] = True
        try:
            response = self.client.chat.completions.create(**kwargs)
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            yield f"\n[LLM Stream Error: {e}]"


class LLMFactory:
    _registry: Dict[str, type[BaseLLM]] = {}

    @classmethod
    def register(cls, provider: str, llm_class: type[BaseLLM]):
        cls._registry[provider] = llm_class

    @classmethod
    def create(cls, config: LLMConfig, validate: bool = True) -> BaseLLM:
        if config.provider not in cls._registry:
            raise ValueError(f"Unknown provider '{config.provider}'")

        llm_class = cls._registry[config.provider]
        client = llm_class(config)

        if validate:
            try:
                client.generate("", "ping")
            except Exception as e:
                raise RuntimeError(
                    f"Model validation failed for LLM '{config.model_name}': {str(e)}"
                )

        return client


LLMFactory.register("openai", StandardLLM)
LLMFactory.register("ollama", StandardLLM)
