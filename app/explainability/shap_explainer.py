"""SHAP-based local explainability for the Random Forest pipeline.

Pipeline structure assumed:
  rf_model (Pipeline)
    ├── "preprocessor"  ← build_preprocessor() result (Pipeline)
    │     └── "preprocessor"  ← ColumnTransformer
    └── "model"         ← RandomForestRegressor

SHAP values are in original dollar units because RF uses no target transformation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import shap as _shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False

# Labels for features derived inside the pipeline (not in FEATURE_COLUMNS)
_DERIVED_LABELS: dict[str, str] = {
    "HouseAge": "Antigüedad de la vivienda (años)",
    "YearsSinceRemodel": "Años desde última remodelación",
    "has_garage": "Tiene garage",
    "has_bsmt": "Tiene sótano",
}


def _feature_label(key: str) -> str:
    if key in _DERIVED_LABELS:
        return _DERIVED_LABELS[key]
    try:
        from app.catalog import FeatureCatalog

        return FeatureCatalog.label(key)
    except Exception:
        return key


class SHAPExplainer:
    """Computes SHAP local attributions for a single prediction from an RF pipeline."""

    def __init__(self) -> None:
        self._explainer: object | None = None
        self._model_id: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return _SHAP_AVAILABLE

    def compute_factors(
        self, sample: pd.DataFrame, rf_model, top_n: int = 5
    ) -> dict[str, list[dict]]:
        """Return the top positive and negative SHAP factors for one prediction.

        Returns:
            {
                "positive": [{"feature_label": str, "shap": float}, ...],
                "negative": [{"feature_label": str, "shap": float}, ...]
            }
        Gracefully returns empty lists on any error.
        """
        if not _SHAP_AVAILABLE:
            return {"positive": [], "negative": []}

        try:
            outer_pp = rf_model.named_steps["preprocessor"]
            X_transformed = outer_pp.transform(sample)

            # ColumnTransformer is the inner "preprocessor" step
            inner_ct = outer_pp.named_steps["preprocessor"]
            feature_names = list(inner_ct.get_feature_names_out())

            explainer = self._get_explainer(rf_model)
            shap_vals = explainer.shap_values(X_transformed)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            shap_row = np.asarray(shap_vals[0])

            aggregated = self._aggregate(feature_names, shap_row)

            # Sort by absolute contribution, then split pos/neg
            sorted_items = sorted(
                aggregated.items(), key=lambda x: abs(x[1]), reverse=True
            )

            positive: list[dict] = []
            negative: list[dict] = []
            for base, sv in sorted_items:
                if sv > 0 and len(positive) < top_n:
                    positive.append({"feature_label": _feature_label(base), "shap": sv})
                elif sv < 0 and len(negative) < top_n:
                    negative.append({"feature_label": _feature_label(base), "shap": abs(sv)})
                if len(positive) >= top_n and len(negative) >= top_n:
                    break

            return {"positive": positive, "negative": negative}

        except Exception:
            return {"positive": [], "negative": []}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_explainer(self, rf_model):
        model_id = id(rf_model)
        if self._explainer is None or self._model_id != model_id:
            rf_step = rf_model.named_steps["model"]
            self._explainer = _shap.TreeExplainer(rf_step)
            self._model_id = model_id
        return self._explainer

    @staticmethod
    def _aggregate(feature_names: list[str], shap_row: np.ndarray) -> dict[str, float]:
        """Sum SHAP values for multi-column OHE encodings back to the original feature."""
        aggregated: dict[str, float] = {}
        for name, sv in zip(feature_names, shap_row):
            prefix, _, rest = name.partition("__")
            # OHE produces cat__MSZoning_RL  →  base = MSZoning
            base = rest.split("_", 1)[0] if prefix == "cat" else rest
            aggregated[base] = aggregated.get(base, 0.0) + float(sv)
        return aggregated
