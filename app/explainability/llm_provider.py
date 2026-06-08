"""LLM provider abstraction for natural-language explanation generation.

Hierarchy:
  BaseLLMProvider (ABC)
  ├── OllamaProvider   — local Ollama server (primary)
  └── MockProvider     — template-based fallback (always available)

Configuration via environment variables (loaded from .env if python-dotenv is installed):
  LLM_PROVIDER   = "ollama" | "mock"          (default: "ollama")
  OLLAMA_HOST    = "http://localhost:11434"    (default)
  OLLAMA_MODEL   = "qwen3:8b"                 (default; llama3.2:3b for low-spec)
  OLLAMA_TIMEOUT = 30                          (seconds)
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

# Load .env if available (optional dependency)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import requests as _requests

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False


class BaseLLMProvider(ABC):
    """Contract for all LLM backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    def is_available(self) -> bool:
        return True


class OllamaProvider(BaseLLMProvider):
    """Calls a local Ollama instance via its REST API.

    Supports qwen3:8b (default) and llama3.2:3b for lower-spec hardware.
    For Qwen3 models, thinking-mode tokens (<think>…</think>) are stripped
    automatically so the user only sees the final answer.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout: int = 30,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        if not _REQUESTS_AVAILABLE:
            return False
        try:
            resp = _requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str) -> str:
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError("requests is not installed. Run: pip install requests")

        is_qwen3 = "qwen3" in self.model.lower()
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # Disable extended thinking for Qwen3 — cleaner output, faster
            "think": False if is_qwen3 else True,
            "options": {"temperature": 0.3, "num_predict": 300},
        }

        resp = _requests.post(
            f"{self.host}/api/generate", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        text = resp.json()["response"].strip()

        # Strip any residual <think>…</think> blocks (safety net)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text


class MockProvider(BaseLLMProvider):
    """Template-based fallback when Ollama is unavailable.

    Parses the factor blocks already embedded in the prompt so the explanation
    mentions the actual drivers instead of generic text.
    """

    def generate(self, prompt: str) -> str:
        price_match = re.search(r"\$([0-9,]+)", prompt)
        price_str = price_match.group(0) if price_match else "el valor estimado"

        pos_factors = self._parse_block(prompt, "AUMENTAN")
        neg_factors = self._parse_block(prompt, "REDUCEN")

        if pos_factors:
            pos_text = ", ".join(f"**{f}**" for f in pos_factors[:3])
            pos_sentence = f"Los factores que más elevan su valor son {pos_text}."
        else:
            pos_sentence = "Su valor refleja la calidad general y el tamaño de la propiedad."

        if neg_factors:
            neg_text = " y ".join(f"**{f}**" for f in neg_factors[:2])
            neg_sentence = f"Los aspectos que moderan el precio a la baja incluyen {neg_text}."
        else:
            neg_sentence = ""

        footer = (
            "\n\n_Explicación automática — activa Ollama (`ollama serve`) "
            "para obtener un análisis con lenguaje natural más detallado._"
        )

        return f"Esta vivienda tiene un precio estimado de **{price_str}**. {pos_sentence} {neg_sentence}{footer}".strip()

    @staticmethod
    def _parse_block(prompt: str, keyword: str) -> list[str]:
        match = re.search(
            rf"Factores que {keyword}.*?:(.*?)(?:Factores que|Instrucciones:|$)",
            prompt,
            re.DOTALL,
        )
        if not match:
            return []
        lines = match.group(1).strip().splitlines()
        return [line.lstrip("- ").strip() for line in lines if line.strip().startswith("-")]


def build_provider(
    provider_type: str | None = None,
    ollama_host: str | None = None,
    ollama_model: str | None = None,
    ollama_timeout: int | None = None,
) -> BaseLLMProvider:
    """Instantiate the configured LLM provider.

    Explicit arguments take precedence over environment variables.
    """
    ptype = provider_type or os.getenv("LLM_PROVIDER", "ollama")
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
    raw_timeout = ollama_timeout or os.getenv("OLLAMA_TIMEOUT", "30")
    timeout = int(raw_timeout)

    if ptype == "mock":
        return MockProvider()
    return OllamaProvider(host=host, model=model, timeout=timeout)
