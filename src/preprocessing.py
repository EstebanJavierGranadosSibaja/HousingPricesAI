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
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

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

# Features with skew > 1 verified on train.csv (1460 rows):
#   LotArea=12.21, TotalBsmtSF=1.52, 1stFlrSF=1.38, GrLivArea=1.37.
# StandardScaler centers/rescales but does NOT fix distribution shape;
# log1p compresses the right tail and stabilizes LinearRegression coefficients.
SKEWED_NUMERICS: frozenset[str] = frozenset(
    {"GrLivArea", "LotArea", "1stFlrSF", "TotalBsmtSF"}
)

# Quality variables with natural ordinal scale: Po < Fa < TA < Gd < Ex.
# OrdinalEncoder preserves this monotone relationship; OneHotEncoder discards it.
ORDINAL_QUALITY_FEATURES: list[str] = ["KitchenQual", "ExterQual"]
BSMT_QUAL_FEATURE: str = "BsmtQual"
ORDINAL_QUALITY_ORDER: list[str] = ["Po", "Fa", "TA", "Gd", "Ex"]
# NoBasement is injected by DomainImputer for structurally absent basements.
BSMT_QUAL_ORDER: list[str] = ["NoBasement", "Po", "Fa", "TA", "Gd", "Ex"]

# YearBuilt/YearRemodAdd are consumed by AgeFeatureCreator and replaced by
# HouseAge/YearsSinceRemodel, which are more interpretable and avoid
# collinearity with the derived features in the LinearRegression.
_YEAR_INPUTS: frozenset[str] = frozenset({"YearBuilt", "YearRemodAdd"})


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
    """Aplica reglas de dominio para nulos estructurales y crea flags (has_garage, has_bsmt).

    Solo opera sobre columnas que pertenecen al set oficial de 22 features
    (BsmtQual, TotalBsmtSF, GarageArea, GarageCars). Se mantiene alineado con el
    DomainImputer del notebook para evitar divergencias entre evidencia y produccion.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Basement: BsmtQual NA es un nulo ESTRUCTURAL (la vivienda no tiene
        # sotano) -> etiqueta explicita y area de sotano en 0.
        if "BsmtQual" in X.columns:
            if X["BsmtQual"].dtype != "object":
                X["BsmtQual"] = X["BsmtQual"].astype("object")
            mask = X["BsmtQual"].isna()
            X.loc[mask, "BsmtQual"] = "NoBasement"
            if "TotalBsmtSF" in X.columns:
                X.loc[mask, "TotalBsmtSF"] = 0

        # Garage: las variables numericas de garaje pertenecen al set de 22
        # features; un faltante se interpreta como ausencia de garaje (0).
        if "GarageArea" in X.columns:
            X["GarageArea"] = pd.to_numeric(X["GarageArea"], errors="coerce").fillna(0)
        if "GarageCars" in X.columns:
            X["GarageCars"] = pd.to_numeric(X["GarageCars"], errors="coerce").fillna(0)

        # Flags binarios: senal explicita "tiene / no tiene" + explicabilidad.
        if "GarageArea" in X.columns:
            X["has_garage"] = (X["GarageArea"] > 0).astype(int)
        elif "GarageCars" in X.columns:
            X["has_garage"] = (X["GarageCars"] > 0).astype(int)
        else:
            X["has_garage"] = 0

        if "TotalBsmtSF" in X.columns:
            X["has_bsmt"] = (pd.to_numeric(X["TotalBsmtSF"], errors="coerce").fillna(0) > 0).astype(int)
        else:
            X["has_bsmt"] = 0

        return X


class AgeFeatureCreator(BaseEstimator, TransformerMixin):
    """Derives HouseAge and YearsSinceRemodel from YearBuilt/YearRemodAdd.

    The reference year is learned from training data (max observed year) to
    avoid leakage across CV folds. For the Ames dataset this yields 2010.

    YearBuilt and YearRemodAdd remain in X after transform; they are dropped
    downstream by ColumnTransformer(remainder='drop') since they appear in
    no transformer pipeline. This avoids perfect collinearity with the
    derived features inside LinearRegression.
    """

    def __init__(self, reference_year: int | None = None):
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y=None):
        if self.reference_year is not None:
            self.reference_year_ = self.reference_year
        else:
            ref = 1900
            for col in ("YearBuilt", "YearRemodAdd"):
                if col in X.columns:
                    val = pd.to_numeric(X[col], errors="coerce").dropna()
                    if not val.empty:
                        ref = max(ref, int(val.max()))
            self.reference_year_ = ref
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "YearBuilt" in X.columns:
            X["HouseAge"] = self.reference_year_ - pd.to_numeric(
                X["YearBuilt"], errors="coerce"
            )
        if "YearRemodAdd" in X.columns:
            X["YearsSinceRemodel"] = self.reference_year_ - pd.to_numeric(
                X["YearRemodAdd"], errors="coerce"
            )
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
            numeric = pd.to_numeric(X[col], errors="coerce").astype("float64")
            X[col] = numeric.clip(lower=lo, upper=hi)
        return X


def build_preprocessor(
    scale_numeric: bool = True,
    features: list[str] | None = None,
    rare_min_freq: float = 0.01,
    winsorize: bool = True,
    reference_year: int | None = None,
):
    """Construye Pipeline: DomainImputer -> AgeFeatureCreator -> ColumnTransformer.

    Mejoras respecto a la version anterior:
    - log1p en variables con skew > 1 (LotArea, GrLivArea, 1stFlrSF, TotalBsmtSF)
      antes del StandardScaler; corrige asimetria que el scaler no puede corregir.
    - OrdinalEncoder (Po<Fa<TA<Gd<Ex) para KitchenQual, ExterQual y BsmtQual
      en lugar de OneHotEncoder; preserva el orden natural de calidad.
    - AgeFeatureCreator reemplaza YearBuilt/YearRemodAdd por HouseAge/YearsSinceRemodel;
      evita colinealidad perfecta y mejora interpretabilidad.
    - Flags binarias (has_garage, has_bsmt) en passthrough: no requieren log ni scaler.
    - scale_numeric=True aplica StandardScaler solo en LR; RF es invariante a escala.
    """
    features = features or load_features()

    # Numeric candidates (YearBuilt/YearRemodAdd excluidos; reemplazados por age features)
    numeric_candidates = {
        "1stFlrSF",
        "2ndFlrSF",
        "BsmtFinSF1",
        "BsmtUnfSF",
        "Fireplaces",
        "GrLivArea",
        "LotArea",
        "LotFrontage",
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

    # Split by skewness: log1p only for features with skew > 1 (verified on data)
    skewed_features = [c for c in numeric_raw_features if c in SKEWED_NUMERICS]
    regular_raw_features = [c for c in numeric_raw_features if c not in SKEWED_NUMERICS]

    # Age features derived from YearBuilt/YearRemodAdd (skew ~0.5-0.6, no log needed)
    age_features: list[str] = []
    if "YearBuilt" in features:
        age_features.append("HouseAge")
    if "YearRemodAdd" in features:
        age_features.append("YearsSinceRemodel")

    regular_all = regular_raw_features + age_features

    # Binary flags: 0/1, passthrough (no log, no scaler needed)
    binary_features = ["has_garage", "has_bsmt"]

    # Ordinal quality features (string categories with natural order)
    ordinal_features = [c for c in ORDINAL_QUALITY_FEATURES if c in features]
    bsmt_qual = BSMT_QUAL_FEATURE if BSMT_QUAL_FEATURE in features else None

    # Truly nominal categoricals (excluded: numerics, age, binary, ordinals, year inputs)
    all_handled = (
        set(numeric_raw_features)
        | set(age_features)
        | set(binary_features)
        | set(ordinal_features)
        | ({bsmt_qual} if bsmt_qual else set())
        | _YEAR_INPUTS
    )
    categorical_features = [c for c in features if c not in all_handled]

    # --- Sub-pipelines ---

    # Skewed numerics: clip -> log1p -> optional StandardScaler
    skewed_steps: list[tuple] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
    ]
    if scale_numeric:
        skewed_steps.append(("scaler", StandardScaler()))

    # Regular numerics + age features: clip -> optional StandardScaler
    regular_steps: list[tuple] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        regular_steps.append(("scaler", StandardScaler()))

    # Ordinal quality (KitchenQual, ExterQual): impute TA -> OrdinalEncoder -> optional scaler
    ordinal_cats = [ORDINAL_QUALITY_ORDER] * len(ordinal_features)
    ordinal_steps: list[tuple] = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="TA")),
        (
            "ord",
            OrdinalEncoder(
                categories=ordinal_cats,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
    if scale_numeric:
        ordinal_steps.append(("scaler", StandardScaler()))

    # BsmtQual: impute NoBasement -> OrdinalEncoder (NoBasement=0 is lowest) -> optional scaler
    bsmt_steps: list[tuple] = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="NoBasement")),
        (
            "ord",
            OrdinalEncoder(
                categories=[BSMT_QUAL_ORDER],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ]
    if scale_numeric:
        bsmt_steps.append(("scaler", StandardScaler()))

    # Nominal categoricals: impute Missing -> OneHotEncoder
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # --- ColumnTransformer ---
    transformers: list[tuple] = []
    if skewed_features:
        transformers.append(("skewed", Pipeline(skewed_steps), skewed_features))
    if regular_all:
        transformers.append(("num", Pipeline(regular_steps), regular_all))
    transformers.append(("binary", "passthrough", binary_features))
    if ordinal_features:
        transformers.append(("ordinal", Pipeline(ordinal_steps), ordinal_features))
    if bsmt_qual:
        transformers.append(("bsmt_qual", Pipeline(bsmt_steps), [bsmt_qual]))
    if categorical_features:
        transformers.append(("cat", cat_pipeline, categorical_features))

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    # --- Full pipeline ---
    # QuantileClipper clips raw numerics + age features (binary flags excluded: 0/1)
    all_numeric_for_clip = numeric_raw_features + age_features

    pipeline_steps: list[tuple] = [
        ("domain", DomainImputer()),
        ("age", AgeFeatureCreator(reference_year=reference_year)),
        ("rare", RareCategoryGrouper(columns=categorical_features, min_freq=rare_min_freq)),
    ]

    if winsorize:
        pipeline_steps.append(
            (
                "winsor",
                QuantileClipper(columns=all_numeric_for_clip, lower_q=0.01, upper_q=0.99),
            )
        )

    pipeline_steps.append(("preprocessor", ct))
    return Pipeline(pipeline_steps)


def build_prediction_frame_from_dict(values: dict) -> pd.DataFrame:
    """Crea DataFrame con orden del manifest; rellena faltantes con NA."""
    features = load_features()
    df = pd.DataFrame([values])
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    return df[features]
