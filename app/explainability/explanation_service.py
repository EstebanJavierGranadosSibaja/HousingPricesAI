"""Orchestrates SHAP attribution + LLM generation into a single explanation call.

Flow:
  ExplanationService.explain(sample, rf_model, predicted_price)
    ├── SHAPExplainer.compute_factors()   →  top positive / negative features
    ├── _build_prompt()                   →  structured Spanish prompt
    ├── OllamaProvider.generate()         →  LLM natural-language explanation
    │   └── [fallback] MockProvider       →  template explanation if Ollama down
    └── return (explanation_text, factors_dict)
"""

from __future__ import annotations

import pandas as pd

from app.explainability.llm_provider import BaseLLMProvider, MockProvider, build_provider
from app.explainability.shap_explainer import SHAPExplainer

_PROMPT_TEMPLATE = """\
Eres un asesor inmobiliario profesional. Genera una explicación clara y breve para un comprador de vivienda.

Precio estimado: ${price:,.0f}

Factores que AUMENTAN el valor de la vivienda:
{positive_block}

Factores que REDUCEN el valor de la vivienda:
{negative_block}

Instrucciones:
- Explica en español claro y amigable por qué la vivienda tiene ese precio estimado.
- Máximo 150 palabras.
- Usa lenguaje sencillo para personas sin conocimientos técnicos.
- No menciones "SHAP", "modelo", "machine learning", ni ningún término técnico.
- No inventes información adicional a lo que se te provee.
- Comienza directamente con la explicación, sin saludos ni introducciones.
"""


class ExplanationService:
    """Generates natural-language price explanations using SHAP + LLM."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        shap_explainer: SHAPExplainer | None = None,
    ) -> None:
        self._llm = llm_provider or build_provider()
        self._shap = shap_explainer or SHAPExplainer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self, sample: pd.DataFrame, rf_model, predicted_price: float
    ) -> tuple[str, dict]:
        """Generate a natural-language explanation for the predicted price.

        Returns:
            (explanation_text, factors_dict)
            factors_dict = {"positive": [...], "negative": [...], "error": str|None}
        """
        factors = self._shap.compute_factors(sample, rf_model, top_n=5)
        if not factors.get("positive") and not factors.get("negative"):
            factors = self._importance_fallback(rf_model)
        text = self._generate_text(predicted_price, factors)
        return text, factors

    def llm_status(self) -> dict:
        """Return a dict describing LLM availability for UI display."""
        available = self._llm.is_available()
        model_name = getattr(self._llm, "model", "plantilla")
        provider_name = type(self._llm).__name__
        return {
            "available": available,
            "provider": provider_name,
            "model": model_name,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_text(self, price: float, factors: dict) -> str:
        pos = factors.get("positive", [])
        neg = factors.get("negative", [])

        if not pos and not neg:
            return self._no_factors_text(price)

        pos_block = (
            "\n".join(f"- {f['feature_label']}" for f in pos)
            or "- (sin factores positivos destacados)"
        )
        neg_block = (
            "\n".join(f"- {f['feature_label']}" for f in neg)
            or "- (sin factores negativos destacados)"
        )

        prompt = _PROMPT_TEMPLATE.format(
            price=price,
            positive_block=pos_block,
            negative_block=neg_block,
        )

        provider = self._llm
        try:
            if not provider.is_available():
                provider = MockProvider()
            return provider.generate(prompt)
        except Exception:
            return MockProvider().generate(prompt)

    @staticmethod
    def _importance_fallback(rf_model) -> dict:
        """Build synthetic pos/neg factor list from global feature importances."""
        try:
            from app.analytics import PredictionAnalytics
            imp = PredictionAnalytics.feature_importances(rf_model, top_n=6)
            if imp is None or imp.empty:
                return {"positive": [], "negative": [], "error": "no importances"}
            positive = [
                {"feature_label": row["Variable"], "shap": row["Importancia"]}
                for _, row in imp.head(3).iterrows()
            ]
            return {"positive": positive, "negative": [], "error": "shap_fallback"}
        except Exception as exc:
            return {"positive": [], "negative": [], "error": str(exc)}

    @staticmethod
    def _no_factors_text(price: float) -> str:
        return (
            f"El precio estimado de la vivienda es **${price:,.0f}**. "
            "No fue posible calcular los factores individuales para esta predicción. "
            "La estimación se basa en los patrones aprendidos del mercado inmobiliario local."
        )
