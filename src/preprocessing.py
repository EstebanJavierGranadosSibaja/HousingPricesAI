"""Preprocessing utilities for the housing price project."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_MANIFEST_PATH = PROJECT_ROOT / "data" / "features.json"

# Canonical 22-feature set used across notebook, training and inference.
FEATURE_COLUMNS = [
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "OverallQual",
    "OverallCond",
    "GarageCars",
    "GarageArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "MSZoning",
    "Neighborhood",
    "KitchenQual",
    "BsmtQual",
    "ExterQual",
    "Foundation",
]

TARGET_COLUMN = "SalePrice"


def load_features(manifest_path: Path | None = None) -> list[str]:
    path = manifest_path or FEATURES_MANIFEST_PATH
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("features") or data.get("columns") or FEATURE_COLUMNS
            if isinstance(data, list):
                return data
        except Exception:
            return FEATURE_COLUMNS
    return FEATURE_COLUMNS


def required_columns() -> list[str]:
    return load_features() + [TARGET_COLUMN]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def get_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extrae X,y usando el manifest de features (o FEATURE_COLUMNS fallback)."""
    validate_required_columns(df)
    features = load_features()
    data = df[features + [TARGET_COLUMN]].copy()
    data = data.dropna(subset=[TARGET_COLUMN])
    x = data[features]
    y = data[TARGET_COLUMN]
    return x, y


class DomainImputer(BaseEstimator, TransformerMixin):
    """Aplica reglas de dominio: NA->'NoX' y crea flags (has_garage, has_bsmt, has_masonry)."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        def ensure_object_dtype(column: str) -> None:
            if column in X.columns and X[column].dtype != "object":
                X[column] = X[column].astype("object")

        # MasVnr: si no hay tipo, normalmente MasVnrArea==0 -> marcar como 'NoMasonry'
        if "MasVnrType" in X.columns:
            ensure_object_dtype("MasVnrType")
            mask = X["MasVnrType"].isna()
            X.loc[mask, "MasVnrType"] = "NoMasonry"
            if "MasVnrArea" in X.columns:
                X.loc[mask, "MasVnrArea"] = 0

        # Garage: si no hay GarageType -> no garage
        if "GarageType" in X.columns:
            ensure_object_dtype("GarageType")
            mask = X["GarageType"].isna()
            X.loc[mask, "GarageType"] = "NoGarage"
            if "GarageArea" in X.columns:
                X.loc[mask, "GarageArea"] = 0
            if "GarageCars" in X.columns:
                X.loc[mask, "GarageCars"] = 0
            if "GarageFinish" in X.columns:
                ensure_object_dtype("GarageFinish")
                X.loc[mask, "GarageFinish"] = "NoGarage"

        # Basement: si BsmtQual NA -> no basement
        if "BsmtQual" in X.columns:
            ensure_object_dtype("BsmtQual")
            mask = X["BsmtQual"].isna()
            X.loc[mask, "BsmtQual"] = "NoBasement"
            if "TotalBsmtSF" in X.columns:
                X.loc[mask, "TotalBsmtSF"] = 0

        # Flags binarios (utiles para modelos y explicabilidad).
        # Siempre se crean para evitar fallos cuando el manifest cambia.
        if "MasVnrArea" in X.columns:
            X["has_masonry"] = (X["MasVnrArea"].fillna(0) > 0).astype(int)
        else:
            X["has_masonry"] = 0

        if "GarageArea" in X.columns:
            X["has_garage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
        elif "GarageCars" in X.columns:
            X["has_garage"] = (pd.to_numeric(X["GarageCars"], errors="coerce").fillna(0) > 0).astype(int)
        else:
            X["has_garage"] = 0

        if "TotalBsmtSF" in X.columns:
            X["has_bsmt"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)
        else:
            X["has_bsmt"] = 0

        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group rare categories under a shared label.

    Rare means category frequency lower than min_freq in the training split.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        min_freq: float = 0.01,
        other_label: str = "Other",
    ):
        self.columns = columns
        self.min_freq = min_freq
        self.other_label = other_label
        self.rare_categories_: dict[str, set[str]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or [c for c in X.columns if X[c].dtype == "object"]
        self.rare_categories_ = {}

        for col in cols:
            if col not in X.columns:
                continue
            values = X[col].dropna().astype(str)
            if values.empty:
                self.rare_categories_[col] = set()
                continue
            freq = values.value_counts(normalize=True)
            rare = set(freq[freq < self.min_freq].index.tolist())
            self.rare_categories_[col] = rare
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, rare_values in self.rare_categories_.items():
            if col not in X.columns or not rare_values:
                continue
            X[col] = X[col].apply(
                lambda v: self.other_label if pd.notna(v) and str(v) in rare_values else v
            )
        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric outliers to configurable quantile bounds."""

    def __init__(
        self,
        columns: list[str] | None = None,
        lower_q: float = 0.01,
        upper_q: float = 0.99,
    ):
        self.columns = columns
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.bounds_ = {}
        for col in cols:
            if col not in X.columns:
                continue
            series = pd.to_numeric(X[col], errors="coerce").dropna()
            if series.empty:
                continue
            lo = float(series.quantile(self.lower_q))
            hi = float(series.quantile(self.upper_q))
            self.bounds_[col] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col not in X.columns:
                continue
            numeric = pd.to_numeric(X[col], errors="coerce")
            X[col] = numeric.clip(lower=lo, upper=hi)
        return X


def build_preprocessor(
    scale_numeric: bool = True,
    features: list[str] | None = None,
    rare_min_freq: float = 0.01,
    winsorize: bool = True,
):
    """Construye Pipeline: DomainImputer -> ColumnTransformer.

    - scale_numeric=True añade StandardScaler (útil para LinearRegression).
    """
    features = features or load_features()
    numeric_candidates = {
        "1stFlrSF",
        "2ndFlrSF",
        "BsmtFinSF1",
        "BsmtUnfSF",
        "Fireplaces",
        "GrLivArea",
        "LotArea",
        "LotFrontage",
        "YearBuilt",
        "YearRemodAdd",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd",
        "GarageCars",
        "GarageArea",
        "GarageYrBlt",
        "OpenPorchSF",
        "OverallCond",
        "OverallQual",
        "MasVnrArea",
        "TotalBsmtSF",
        "WoodDeckSF",
    }

    numeric_raw_features = [c for c in features if c in numeric_candidates]
    numeric_features = list(numeric_raw_features)
    # añadir flags creados por DomainImputer al bloque numérico
    for flag in ("has_masonry", "has_garage", "has_bsmt"):
        if flag not in numeric_features:
            numeric_features.append(flag)

    categorical_features = [c for c in features if c not in numeric_features]

    # Pipelines
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    pipeline_steps: list[tuple[str, object]] = [
        ("domain", DomainImputer()),
        (
            "rare",
            RareCategoryGrouper(columns=categorical_features, min_freq=rare_min_freq),
        ),
    ]

    if winsorize:
        pipeline_steps.append(
            (
                "winsor",
                QuantileClipper(columns=numeric_raw_features, lower_q=0.01, upper_q=0.99),
            )
        )

    pipeline_steps.append(("preprocessor", ct))
    pipeline = Pipeline(pipeline_steps)
    return pipeline


def build_prediction_frame_from_dict(values: dict) -> pd.DataFrame:
    """Crea DataFrame con orden del manifest; rellena faltantes con NA."""
    features = load_features()
    df = pd.DataFrame([values])
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    return df[features]


def build_prediction_frame(size_m2: float, location: str, rooms: int) -> pd.DataFrame:
    """Compat wrapper for CLI/Streamlit using 3 user inputs from the prompt."""
    values = {
        "GrLivArea": size_m2,
        "Neighborhood": location,
        "BedroomAbvGr": rooms,
    }
    return build_prediction_frame_from_dict(values)

