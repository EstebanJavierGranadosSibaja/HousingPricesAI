"""Preprocessing helpers using a ColumnTransformer pipeline.

This module provides utilities to build a robust preprocessing pipeline that
handles numeric and categorical features. The main functions are:

- ``preprocess_for_training``: fits a preprocessor on the provided dataframe
  and returns the transformed features, target, and the fitted preprocessor.
- ``preprocess_for_inference``: applies a fitted preprocessor to new data.

The pipeline imputes numeric features (median) and scales them, and imputes
categorical features and one-hot encodes them (unknown categories are ignored).
"""
from typing import Tuple, List
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create a ColumnTransformer for the dataframe X.

    Returns: (preprocessor, num_cols, cat_cols)
    """
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(('num', num_transformer, num_cols))
    if cat_cols:
        transformers.append(('cat', cat_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, num_cols, cat_cols


def preprocess_for_training(df: pd.DataFrame, target: str = 'SalePrice') -> Tuple:
    """Fit preprocessor on df and return transformed X (numpy), y (Series) and preprocessor.

    Returns: (X_transformed, y, preprocessor, num_cols, cat_cols)
    """
    df = df.copy()
    y = None
    if target in df.columns:
        y = df[target]
        X = df.drop(columns=[target])
    else:
        X = df

    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor, num_cols, cat_cols


def preprocess_for_inference(preprocessor: ColumnTransformer, df: pd.DataFrame):
    """Apply a fitted preprocessor to a dataframe and return the transformed array."""
    X = df.copy()
    return preprocessor.transform(X)
