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
    def disagreement_level(
        pred_linear: float,
        pred_rf: float,
        metrics_df: pd.DataFrame | None,
    ) -> tuple[str, str, float, float]:
        diff_abs = abs(pred_linear - pred_rf)
        mean_pred = max((pred_linear + pred_rf) / 2.0, 1.0)
        ratio = diff_abs / mean_pred

        rmse_ref = 0.0
        if metrics_df is not None and not metrics_df.empty and "RMSE" in metrics_df.columns:
            rmse_ref = float(metrics_df["RMSE"].mean())

        if ratio < 0.10 and (rmse_ref == 0.0 or diff_abs <= 1.5 * rmse_ref):
            return "baja", "La diferencia esta dentro del rango esperado entre modelos.", diff_abs, ratio
        if ratio < 0.25 and (rmse_ref == 0.0 or diff_abs <= 3.0 * rmse_ref):
            return "media", "Hay diferencia moderada, conviene revisar valores de entrada.", diff_abs, ratio
        return "alta", "La diferencia es alta, puede haber combinaciones atipicas o alta no linealidad.", diff_abs, ratio

    @staticmethod
    def percentile_rank(series: pd.Series, value: float) -> float | None:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return None
        return float((values <= value).mean() * 100.0)

    @staticmethod
    def detect_extreme_inputs(sample: pd.DataFrame, ref_df: pd.DataFrame | None) -> list[str]:
        if ref_df is None or sample.empty:
            return []

        watch_features = ["GrLivArea", "LotArea", "OverallQual", "GarageArea", "YearBuilt"]
        row = sample.iloc[0]
        notes: list[str] = []

        for feature in watch_features:
            if feature not in row.index or feature not in ref_df.columns:
                continue

            value = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue

            p = PredictionAnalytics.percentile_rank(ref_df[feature], float(value))
            if p is None:
                continue

            if p <= 5.0 or p >= 95.0:
                notes.append(f"{FeatureCatalog.label(feature)} esta en percentil {p:.1f}%")

        return notes
