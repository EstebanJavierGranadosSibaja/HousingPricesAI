"""Feature naming helpers and human-readable mappings."""

from __future__ import annotations

from app.config import FEATURE_METADATA


class FeatureCatalog:
    """Provides user-friendly labels and descriptions for technical feature names."""

    @staticmethod
    def label(feature: str, include_code: bool = False) -> str:
        label = str(FEATURE_METADATA.get(feature, {}).get("label", feature))
        if include_code:
            return f"{label} [{feature}]"
        return label

    @staticmethod
    def help_text(feature: str) -> str | None:
        help_value = FEATURE_METADATA.get(feature, {}).get("help")
        if not help_value:
            return None
        return str(help_value)
