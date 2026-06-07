"""Explainability layer: SHAP local attribution + LLM natural-language generation."""

from app.explainability.explanation_service import ExplanationService
from app.explainability.llm_provider import BaseLLMProvider, MockProvider, OllamaProvider
from app.explainability.shap_explainer import SHAPExplainer

__all__ = [
    "ExplanationService",
    "BaseLLMProvider",
    "MockProvider",
    "OllamaProvider",
    "SHAPExplainer",
]
