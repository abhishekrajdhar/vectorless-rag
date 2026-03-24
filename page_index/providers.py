from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from . import llm


class LLMProvider:
    """Abstract LLM provider interface."""

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError()

    async def generate_async(self, prompt: str, **kwargs) -> str:
        # Default sync wrapper
        return self.generate(prompt, **kwargs)


class MockProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return llm._mock_response(prompt)

    async def generate_async(self, prompt: str, **kwargs) -> str:
        return llm._mock_response(prompt)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate(self, prompt: str, model: str = "gpt-4o", **kwargs) -> str:
        return llm.ChatGPT_API(model=model, prompt=prompt, api_key=self.api_key)

    async def generate_async(self, prompt: str, model: str = "gpt-4o", **kwargs) -> str:
        return await llm.ChatGPT_API_async(model=model, prompt=prompt, api_key=self.api_key)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, model: str = None, **kwargs) -> str:
        use_model = model or self.model
        # re-use llm.ChatGPT_API_with_finish_reason which supports gemini provider
        content, _ = llm.ChatGPT_API_with_finish_reason(model=use_model or '', prompt=prompt, api_key=self.api_key)
        return content

    async def generate_async(self, prompt: str, model: str = None, **kwargs) -> str:
        # The llm.async supports mock and openai; Gemini is implemented sync only.
        return self.generate(prompt, model=model)


def get_provider(name: str, **kwargs) -> LLMProvider:
    name = (name or "").lower()
    if name == "openai":
        return OpenAIProvider(api_key=kwargs.get("api_key"))
    if name == "gemini":
        return GeminiProvider(api_key=kwargs.get("api_key"), model=kwargs.get("model"))
    return MockProvider()
