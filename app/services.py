"""Data access and input preparation services for the Streamlit app."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from app.catalog import FeatureCatalog
from app.config import (
    DATA_PATH,
    FEATURE_GROUP_BLUEPRINT,
    METRICS_PATH,
    MODELS_DIR,
    NUMERIC_FEATURE_HINTS,
)


class DataRepository:
    """Loads persisted artifacts used by the app."""

    @staticmethod
    @st.cache_resource
    def load_models():
        linear_path = MODELS_DIR / "linear_regression.joblib"
        rf_path = MODELS_DIR / "random_forest.joblib"

        if not linear_path.exists() or not rf_path.exists():
            return None, None

        return joblib.load(linear_path), joblib.load(rf_path)

    @staticmethod
    @st.cache_data
    def load_reference_data() -> pd.DataFrame | None:
        if not DATA_PATH.exists():
            return None
        return pd.read_csv(DATA_PATH)

    @staticmethod
    @st.cache_data
    def load_metrics() -> pd.DataFrame | None:
        if not METRICS_PATH.exists():
            return None
        df = pd.read_csv(METRICS_PATH)
        if "RMSE" in df.columns:
            df = df.sort_values(by="RMSE", ascending=True)
        return df


class InputPreparationService:
    """Transforms raw user inputs into model-ready values."""

    @staticmethod
    def numeric_default(series: pd.Series) -> float:
        values = pd.to_numeric(series, errors="coerce")
        if values.notna().any():
            return float(values.median())
        return 0.0

    @staticmethod
    def categorical_default(series: pd.Series) -> str:
        values = series.dropna().astype(str)
        if not values.empty:
            return str(values.mode().iloc[0])
        return "Missing"

    @staticmethod
    def numeric_bounds(series: pd.Series) -> tuple[float, float]:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return 0.0, 100.0
        low = float(values.quantile(0.01))
        high = float(values.quantile(0.99))
        if low == high:
            high = low + 1.0
        return low, high

    @staticmethod
    def series_quantile(ref_df: pd.DataFrame | None, column: str, q: float, fallback: float) -> float:
        if ref_df is None or column not in ref_df.columns:
            return fallback
        values = pd.to_numeric(ref_df[column], errors="coerce").dropna()
        if values.empty:
            return fallback
        return float(values.quantile(q))

    @staticmethod
    def series_mode(ref_df: pd.DataFrame | None, column: str, fallback: str, rank: int = 0) -> str:
        if ref_df is None or column not in ref_df.columns:
            return fallback
        counts = ref_df[column].dropna().astype(str).value_counts()
        if counts.empty:
            return fallback
        rank = min(rank, len(counts) - 1)
        return str(counts.index[rank])

    @staticmethod
    def to_float(value: object, fallback: float = 0.0) -> float:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(parsed):
            return fallback
        return float(parsed)

    @staticmethod
    def clip_numeric_feature(feature: str, value: float, ref_df: pd.DataFrame | None) -> float:
        if ref_df is not None and feature in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[feature]):
            low, high = InputPreparationService.numeric_bounds(ref_df[feature])
            value = max(low, min(value, high))
        return value

    @staticmethod
    def build_defaults(features: list[str], ref_df: pd.DataFrame | None) -> dict[str, object]:
        defaults: dict[str, object] = {}
        for feature in features:
            if ref_df is not None and feature in ref_df.columns:
                series = ref_df[feature]
                if pd.api.types.is_numeric_dtype(series):
                    defaults[feature] = InputPreparationService.numeric_default(series)
                else:
                    defaults[feature] = InputPreparationService.categorical_default(series)
            else:
                defaults[feature] = 0.0 if feature in NUMERIC_FEATURE_HINTS else "Missing"
        return defaults

    @staticmethod
    def apply_preset(
        preset: str,
        defaults: dict[str, object],
        ref_df: pd.DataFrame | None,
    ) -> dict[str, object]:
        values = dict(defaults)

        if preset == "Base (mediana/moda)" or ref_df is None:
            return values

        if preset == "Casa compacta":
            values["GrLivArea"] = InputPreparationService.series_quantile(
                ref_df,
                "GrLivArea",
                0.25,
                float(values.get("GrLivArea", 1200.0)),
            )
            values["LotArea"] = InputPreparationService.series_quantile(
                ref_df,
                "LotArea",
                0.25,
                float(values.get("LotArea", 8000.0)),
            )
            values["OverallQual"] = round(InputPreparationService.series_quantile(ref_df, "OverallQual", 0.35, 5.0))
            values["BedroomAbvGr"] = round(InputPreparationService.series_quantile(ref_df, "BedroomAbvGr", 0.40, 2.0))
            values["GarageCars"] = round(InputPreparationService.series_quantile(ref_df, "GarageCars", 0.35, 1.0))
            values["Neighborhood"] = InputPreparationService.series_mode(
                ref_df,
                "Neighborhood",
                str(values.get("Neighborhood", "NAmes")),
                rank=2,
            )
            return values

        if preset == "Casa familiar":
            values["GrLivArea"] = InputPreparationService.series_quantile(
                ref_df,
                "GrLivArea",
                0.55,
                float(values.get("GrLivArea", 1500.0)),
            )
            values["LotArea"] = InputPreparationService.series_quantile(
                ref_df,
                "LotArea",
                0.55,
                float(values.get("LotArea", 9000.0)),
            )
            values["OverallQual"] = round(InputPreparationService.series_quantile(ref_df, "OverallQual", 0.60, 6.0))
            values["BedroomAbvGr"] = round(InputPreparationService.series_quantile(ref_df, "BedroomAbvGr", 0.65, 3.0))
            values["GarageCars"] = round(InputPreparationService.series_quantile(ref_df, "GarageCars", 0.60, 2.0))
            values["Neighborhood"] = InputPreparationService.series_mode(
                ref_df,
                "Neighborhood",
                str(values.get("Neighborhood", "NAmes")),
                rank=0,
            )
            return values

        if preset == "Casa premium":
            values["GrLivArea"] = InputPreparationService.series_quantile(
                ref_df,
                "GrLivArea",
                0.90,
                float(values.get("GrLivArea", 2300.0)),
            )
            values["LotArea"] = InputPreparationService.series_quantile(
                ref_df,
                "LotArea",
                0.88,
                float(values.get("LotArea", 12000.0)),
            )
            values["OverallQual"] = max(8, round(InputPreparationService.series_quantile(ref_df, "OverallQual", 0.90, 8.0)))
            values["BedroomAbvGr"] = max(4, round(InputPreparationService.series_quantile(ref_df, "BedroomAbvGr", 0.85, 4.0)))
            values["GarageCars"] = max(2, round(InputPreparationService.series_quantile(ref_df, "GarageCars", 0.85, 2.0)))
            values["Neighborhood"] = InputPreparationService.series_mode(
                ref_df,
                "Neighborhood",
                str(values.get("Neighborhood", "NridgHt")),
                rank=1,
            )
            return values

        return values

    @staticmethod
    def resolve_feature_groups(features: list[str]) -> list[tuple[str, list[str]]]:
        groups: list[tuple[str, list[str]]] = []
        used: set[str] = set()

        for title, candidates in FEATURE_GROUP_BLUEPRINT:
            current = [feature for feature in candidates if feature in features and feature not in used]
            if current:
                groups.append((title, current))
                used.update(current)

        remaining = [feature for feature in features if feature not in used]
        if remaining:
            groups.append(("Otros", remaining))

        return groups

    @staticmethod
    def slugify(text: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in text)

    @staticmethod
    def build_ab_scenario(
        base_values: dict[str, object],
        ref_df: pd.DataFrame | None,
        area_delta_pct: float,
        quality_delta: int,
        garage_delta: int,
        remodel_delta_years: int,
    ) -> tuple[dict[str, object], list[str]]:
        updated = dict(base_values)
        changes: list[str] = []

        if "GrLivArea" in updated and area_delta_pct != 0:
            current = InputPreparationService.to_float(updated["GrLivArea"], 0.0)
            target = current * (1.0 + area_delta_pct / 100.0)
            target = InputPreparationService.clip_numeric_feature("GrLivArea", target, ref_df)
            updated["GrLivArea"] = round(target, 1)
            changes.append(f"{FeatureCatalog.label('GrLivArea')}: {current:,.0f} -> {target:,.0f} ft2")

        if "OverallQual" in updated and quality_delta != 0:
            current = InputPreparationService.to_float(updated["OverallQual"], 5.0)
            target = InputPreparationService.clip_numeric_feature("OverallQual", current + quality_delta, ref_df)
            target = int(round(target))
            updated["OverallQual"] = target
            changes.append(f"{FeatureCatalog.label('OverallQual')}: {int(round(current))} -> {target}")

        if "GarageCars" in updated and garage_delta != 0:
            current = InputPreparationService.to_float(updated["GarageCars"], 1.0)
            target = InputPreparationService.clip_numeric_feature("GarageCars", current + garage_delta, ref_df)
            target = int(round(target))
            updated["GarageCars"] = target
            changes.append(f"{FeatureCatalog.label('GarageCars')}: {int(round(current))} -> {target}")

            if "GarageArea" in updated:
                base_area = InputPreparationService.to_float(updated["GarageArea"], 0.0)
                denom = max(current, 1.0)
                scaled_area = base_area * (max(target, 0.0) / denom)
                scaled_area = InputPreparationService.clip_numeric_feature("GarageArea", scaled_area, ref_df)
                updated["GarageArea"] = round(scaled_area, 1)

        if "YearRemodAdd" in updated and remodel_delta_years != 0:
            current = InputPreparationService.to_float(updated["YearRemodAdd"], 2000.0)
            target = InputPreparationService.clip_numeric_feature("YearRemodAdd", current + remodel_delta_years, ref_df)
            target = int(round(target))
            updated["YearRemodAdd"] = target
            changes.append(f"{FeatureCatalog.label('YearRemodAdd')}: {int(round(current))} -> {target}")

        return updated, changes
