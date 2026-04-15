"""Input validation and coercion for stable model inference."""

from __future__ import annotations

import pandas as pd

from app.config import INTEGER_LIKE_FEATURES, NUMERIC_FEATURE_HINTS
from app.services import InputPreparationService


class InputValidator:
    """Coerces and sanitizes user inputs into model-friendly values."""

    @staticmethod
    def _is_numeric_feature(feature: str, ref_df: pd.DataFrame | None) -> bool:
        if feature in NUMERIC_FEATURE_HINTS:
            return True
        if ref_df is not None and feature in ref_df.columns:
            return bool(pd.api.types.is_numeric_dtype(ref_df[feature]))
        return False

    @staticmethod
    def sanitize_values(
        values: dict[str, object],
        features: list[str],
        ref_df: pd.DataFrame | None,
        defaults: dict[str, object],
    ) -> tuple[dict[str, object], list[str]]:
        sanitized: dict[str, object] = {}
        notes: list[str] = []

        for feature in features:
            fallback = defaults.get(feature, 0.0 if feature in NUMERIC_FEATURE_HINTS else "Missing")
            raw_value = values.get(feature, fallback)

            if InputValidator._is_numeric_feature(feature, ref_df):
                numeric_value = InputPreparationService.to_float(raw_value, InputPreparationService.to_float(fallback, 0.0))
                if pd.isna(pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]):
                    notes.append(f"{feature}: se uso valor por defecto numerico por dato invalido.")

                numeric_value = InputPreparationService.clip_numeric_feature(feature, numeric_value, ref_df)
                if feature in INTEGER_LIKE_FEATURES:
                    numeric_value = int(round(numeric_value))
                sanitized[feature] = numeric_value
                continue

            text_value = str(raw_value).strip() if raw_value is not None else ""
            if text_value in {"", "nan", "None"}:
                text_value = str(fallback)
                notes.append(f"{feature}: se uso valor por defecto categorico por dato vacio.")

            if ref_df is not None and feature in ref_df.columns and not pd.api.types.is_numeric_dtype(ref_df[feature]):
                allowed = set(ref_df[feature].dropna().astype(str).unique())
                whitelist = {"Missing", "NoBasement", "NoGarage", "NoMasonry"}
                if allowed and text_value not in allowed and text_value not in whitelist:
                    text_value = str(fallback)
                    notes.append(f"{feature}: categoria no reconocida, se restauro default.")

            sanitized[feature] = text_value

        return sanitized, notes
