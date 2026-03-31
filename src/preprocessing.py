"""Preprocessing utilities for the housing price project."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = ["GrLivArea", "Neighborhood", "BedroomAbvGr"]
TARGET_COLUMN = "SalePrice"


def required_columns() -> list[str]:
    return FEATURE_COLUMNS + [TARGET_COLUMN]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def get_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_required_columns(df)

    data = df[required_columns()].copy()
    data = data.dropna(subset=[TARGET_COLUMN])

    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return x, y


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["GrLivArea", "BedroomAbvGr"]
    categorical_features = ["Neighborhood"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def build_prediction_frame(size_m2: float, location: str, rooms: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GrLivArea": [size_m2],
            "Neighborhood": [location],
            "BedroomAbvGr": [rooms],
        }
    )
