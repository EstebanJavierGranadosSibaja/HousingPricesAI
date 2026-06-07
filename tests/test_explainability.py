"""Unit tests for the explainability layer (SHAP + LLM providers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.explainability.explanation_service import ExplanationService, _PROMPT_TEMPLATE
from app.explainability.llm_provider import MockProvider, OllamaProvider, build_provider
from app.explainability.shap_explainer import SHAPExplainer, _feature_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FailingModel:
    """Dummy model whose named_steps raises KeyError — tests graceful fallback."""
    named_steps: dict = {}


class _MinimalRFModel:
    """Wraps a real sklearn RF to test the SHAPExplainer on actual data."""

    def __init__(self):
        self._built = False
        self._pipeline = None

    def _build(self):
        if self._built:
            return
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        ct = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), ["a", "b", "c"]),
        ], remainder="drop")

        self._pipeline = Pipeline([
            ("preprocessor", Pipeline([
                ("preprocessor", ct),
            ])),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ])

        X = pd.DataFrame({"a": range(50), "b": range(50, 100), "c": range(100, 150)})
        y = pd.Series([float(i * 1000) for i in range(50)])
        self._pipeline.fit(X, y)
        self._built = True

    @property
    def named_steps(self):
        self._build()
        return self._pipeline.named_steps


# ---------------------------------------------------------------------------
# LLM Provider tests
# ---------------------------------------------------------------------------

class TestMockProvider:
    def test_always_available(self):
        assert MockProvider().is_available() is True

    def test_generate_returns_non_empty_string(self):
        result = MockProvider().generate("precio $150,000")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_includes_price_from_prompt(self):
        result = MockProvider().generate("El precio es $175,000 del inmueble.")
        assert "$175,000" in result

    def test_generate_with_any_prompt(self):
        result = MockProvider().generate("test prompt sin precio")
        assert isinstance(result, str)


class TestOllamaProvider:
    def test_reports_unavailable_for_closed_port(self):
        p = OllamaProvider(host="http://localhost:59999", timeout=2)
        assert p.is_available() is False

    def test_generate_raises_on_connection_error(self):
        p = OllamaProvider(host="http://localhost:59999", timeout=2)
        with pytest.raises(Exception):
            p.generate("test")

    def test_default_model_is_qwen3(self):
        p = OllamaProvider()
        assert "qwen3" in p.model

    def test_custom_model_stored(self):
        p = OllamaProvider(model="llama3.2:3b")
        assert p.model == "llama3.2:3b"


class TestBuildProvider:
    def test_explicit_mock(self):
        assert isinstance(build_provider(provider_type="mock"), MockProvider)

    def test_explicit_ollama(self):
        assert isinstance(build_provider(provider_type="ollama"), OllamaProvider)

    def test_env_var_mock(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        assert isinstance(build_provider(), MockProvider)

    def test_env_var_ollama_model(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
        p = build_provider()
        assert isinstance(p, OllamaProvider)
        assert p.model == "llama3.2:3b"

    def test_default_is_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        assert isinstance(build_provider(), OllamaProvider)


# ---------------------------------------------------------------------------
# Feature label mapping
# ---------------------------------------------------------------------------

class TestFeatureLabel:
    def test_house_age_returns_spanish(self):
        label = _feature_label("HouseAge")
        assert "años" in label.lower() or "antigü" in label.lower()

    def test_years_since_remodel_returns_spanish(self):
        label = _feature_label("YearsSinceRemodel")
        assert "años" in label.lower() or "remodel" in label.lower()

    def test_has_garage_returns_spanish(self):
        assert "garage" in _feature_label("has_garage").lower()

    def test_has_bsmt_returns_spanish(self):
        label = _feature_label("has_bsmt")
        assert "sótano" in label.lower() or "sotano" in label.lower()

    def test_known_feature_returns_label(self):
        label = _feature_label("GrLivArea")
        assert isinstance(label, str) and len(label) > 0

    def test_unknown_feature_returns_key(self):
        assert _feature_label("unknown_xyz_feature") == "unknown_xyz_feature"


# ---------------------------------------------------------------------------
# SHAPExplainer structural tests
# ---------------------------------------------------------------------------

class TestSHAPExplainer:
    def test_returns_expected_structure_on_failure(self):
        explainer = SHAPExplainer()
        result = explainer.compute_factors(pd.DataFrame(), _FailingModel(), top_n=5)
        assert set(result.keys()) == {"positive", "negative"}
        assert isinstance(result["positive"], list)
        assert isinstance(result["negative"], list)

    def test_aggregate_correctly_sums_by_prefix(self):
        names = ["skewed__GrLivArea", "num__HouseAge", "cat__MSZoning_RL", "cat__MSZoning_RM"]
        shap_row = np.array([1000.0, -500.0, 200.0, 150.0])
        result = SHAPExplainer._aggregate(names, shap_row)
        assert result["GrLivArea"] == pytest.approx(1000.0)
        assert result["HouseAge"] == pytest.approx(-500.0)
        assert result["MSZoning"] == pytest.approx(350.0)   # 200+150

    def test_aggregate_ordinal_features(self):
        names = ["ordinal__KitchenQual", "bsmt_qual__BsmtQual", "binary__has_garage"]
        shap_row = np.array([800.0, -300.0, 100.0])
        result = SHAPExplainer._aggregate(names, shap_row)
        assert result["KitchenQual"] == pytest.approx(800.0)
        assert result["BsmtQual"] == pytest.approx(-300.0)
        assert result["has_garage"] == pytest.approx(100.0)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("shap"),
        reason="shap not installed"
    )
    def test_real_model_returns_nonempty_factors(self):
        model = _MinimalRFModel()
        explainer = SHAPExplainer()
        sample = pd.DataFrame({"a": [25], "b": [75], "c": [125]})
        result = explainer.compute_factors(sample, model, top_n=3)
        total = len(result["positive"]) + len(result["negative"])
        assert total > 0


# ---------------------------------------------------------------------------
# ExplanationService tests
# ---------------------------------------------------------------------------

class TestExplanationService:
    def test_no_factors_text_contains_price(self):
        text = ExplanationService._no_factors_text(185_000.0)
        assert "185,000" in text or "185000" in text

    def test_explain_with_mock_returns_tuple(self):
        service = ExplanationService(llm_provider=MockProvider())
        text, factors = service.explain(pd.DataFrame(), _FailingModel(), 180_000.0)
        assert isinstance(text, str) and len(text) > 0
        assert isinstance(factors, dict)
        assert "positive" in factors and "negative" in factors

    def test_explain_uses_fallback_when_llm_unavailable(self):
        unavailable = OllamaProvider(host="http://localhost:59999", timeout=2)
        service = ExplanationService(llm_provider=unavailable)
        text, _ = service.explain(pd.DataFrame(), _FailingModel(), 200_000.0)
        assert isinstance(text, str) and len(text) > 0

    def test_prompt_template_has_required_placeholders(self):
        assert "{price" in _PROMPT_TEMPLATE
        assert "{positive_block}" in _PROMPT_TEMPLATE
        assert "{negative_block}" in _PROMPT_TEMPLATE

    def test_llm_status_keys(self):
        service = ExplanationService(llm_provider=MockProvider())
        status = service.llm_status()
        assert {"available", "provider", "model"} == set(status.keys())
        assert status["provider"] == "MockProvider"

    def test_explanation_text_non_empty_with_factors(self):
        factors = {
            "positive": [{"feature_label": "Calidad general", "shap": 5000.0}],
            "negative": [{"feature_label": "Antigüedad", "shap": 2000.0}],
        }
        service = ExplanationService(llm_provider=MockProvider())
        text = service._generate_text(160_000.0, factors)
        assert len(text) > 50
