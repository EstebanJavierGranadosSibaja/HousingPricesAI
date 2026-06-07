"""Prediction analytics helpers used by the Streamlit app."""

from __future__ import annotations

import pandas as pd

from app.catalog import FeatureCatalog


class PredictionAnalytics:
    """Computes model comparison metrics and diagnostics."""

    @staticmethod
    def weighted_consensus(pred_linear: float, pred_rf: float, metrics_df: pd.DataFrame | None) -> float:
        if (
            metrics_df is None
            or metrics_df.empty
            or "Model" not in metrics_df.columns
            or "RMSE" not in metrics_df.columns
        ):
            return (pred_linear + pred_rf) / 2.0

        rmse_map = dict(zip(metrics_df["Model"], metrics_df["RMSE"]))
        rmse_linear = float(rmse_map.get("linear_regression", 1.0))
        rmse_rf = float(rmse_map.get("random_forest", 1.0))
        w_linear = 1.0 / max(rmse_linear, 1e-9)
        w_rf = 1.0 / max(rmse_rf, 1e-9)
        return (pred_linear * w_linear + pred_rf * w_rf) / (w_linear + w_rf)

    @staticmethod
    def price_range(prediction: float, metrics_df: pd.DataFrame | None, model_key: str = "random_forest") -> tuple[float, float]:
        """Estimate a +/- band around the prediction using the model's MAE.

        Falls back to a 7% band when metrics are unavailable.
        """
        margin = prediction * 0.07
        if metrics_df is not None and not metrics_df.empty and {"Model", "MAE"}.issubset(metrics_df.columns):
            mae_map = dict(zip(metrics_df["Model"], metrics_df["MAE"]))
            mae = mae_map.get(model_key)
            if mae is not None and pd.notna(mae):
                margin = float(mae)
        return max(prediction - margin, 0.0), prediction + margin

    @staticmethod
    def feature_importances(rf_model, top_n: int = 8) -> pd.DataFrame | None:
        """Extract Random Forest importances aggregated by original feature.

        Returns a DataFrame with friendly labels, or None if the model does
        not expose importances in the expected pipeline structure.
        """
        try:
            # Soporta modelos envueltos en TransformedTargetRegressor (--log-target).
            pipeline = getattr(rf_model, "regressor_", rf_model)
            inner = pipeline.named_steps["preprocessor"]
            column_transformer = inner.named_steps["preprocessor"]
            names = column_transformer.get_feature_names_out()
            importances = pipeline.named_steps["model"].feature_importances_
        except (AttributeError, KeyError):
            return None

        aggregated: dict[str, float] = {}
        for name, importance in zip(names, importances):
            prefix, _, rest = name.partition("__")
            base = rest.split("_", 1)[0] if prefix == "cat" else rest
            aggregated[base] = aggregated.get(base, 0.0) + float(importance)

        rows = [
            {"Variable": FeatureCatalog.label(base), "Importancia": value}
            for base, value in aggregated.items()
        ]
        result = pd.DataFrame(rows).sort_values("Importancia", ascending=False).head(top_n)
        return result.reset_index(drop=True)

    @staticmethod
    def percentile_rank(series: pd.Series, value: float) -> float | None:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return None
        return float((values <= value).mean() * 100.0)

