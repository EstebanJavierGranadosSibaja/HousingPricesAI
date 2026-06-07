"""Ablation study: quantifies the contribution of each preprocessing improvement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.preprocessing import (
    AgeFeatureCreator,
    DomainImputer,
    QuantileClipper,
    RareCategoryGrouper,
    BSMT_QUAL_ORDER,
    ORDINAL_QUALITY_ORDER,
    build_preprocessor,
    get_training_frame,
)

SKEWED = ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "LotArea"]
REGULAR_WITH_YEAR = [
    "2ndFlrSF", "YearBuilt", "YearRemodAdd", "OverallQual", "OverallCond",
    "GarageCars", "GarageArea", "FullBath", "HalfBath", "BedroomAbvGr",
    "TotRmsAbvGrd", "Fireplaces", "has_garage", "has_bsmt",
]
REGULAR_WITH_AGE = [
    "2ndFlrSF", "OverallQual", "OverallCond", "GarageCars", "GarageArea",
    "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "has_garage", "has_bsmt", "HouseAge", "YearsSinceRemodel",
]
CAT_OHE = ["MSZoning", "Neighborhood", "KitchenQual", "BsmtQual", "ExterQual", "Foundation"]
CAT_NOMINAL = ["MSZoning", "Neighborhood", "Foundation"]
ORDINAL_Q = ["KitchenQual", "ExterQual"]
CLIP_YEAR = SKEWED + ["2ndFlrSF", "YearBuilt", "YearRemodAdd", "OverallQual", "OverallCond",
                      "GarageCars", "GarageArea", "FullBath", "HalfBath", "BedroomAbvGr",
                      "TotRmsAbvGrd", "Fireplaces"]


def make_ohe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def make_ordinal_q():
    return Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="TA")),
        ("ord", OrdinalEncoder(
            categories=[ORDINAL_QUALITY_ORDER] * 2,
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
        ("sc", StandardScaler()),
    ])


def make_bsmt():
    return Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="NoBasement")),
        ("ord", OrdinalEncoder(
            categories=[BSMT_QUAL_ORDER],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
        ("sc", StandardScaler()),
    ])


def make_sk_pipe(with_log: bool, with_scale: bool):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if with_log:
        steps.append(("log", FunctionTransformer(np.log1p)))
    if with_scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps)


def make_reg_pipe(with_scale: bool):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if with_scale:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps)


def build_lr_baseline():
    ct = ColumnTransformer([
        ("num", make_reg_pipe(True), SKEWED + REGULAR_WITH_YEAR),
        ("cat", make_ohe(), CAT_OHE),
    ], remainder="drop")
    return Pipeline([
        ("domain", DomainImputer()),
        ("rare", RareCategoryGrouper(columns=CAT_OHE, min_freq=0.01)),
        ("winsor", QuantileClipper(columns=CLIP_YEAR, lower_q=0.01, upper_q=0.99)),
        ("pp", ct),
        ("model", LinearRegression()),
    ])


def build_lr_with_logtarget():
    return TransformedTargetRegressor(
        regressor=build_lr_baseline(), func=np.log1p, inverse_func=np.expm1
    )


def build_lr_log_features():
    ct = ColumnTransformer([
        ("sk", make_sk_pipe(True, True), SKEWED),
        ("num", make_reg_pipe(True), REGULAR_WITH_YEAR),
        ("cat", make_ohe(), CAT_OHE),
    ], remainder="drop")
    lr_pipe = Pipeline([
        ("domain", DomainImputer()),
        ("rare", RareCategoryGrouper(columns=CAT_OHE, min_freq=0.01)),
        ("winsor", QuantileClipper(columns=CLIP_YEAR, lower_q=0.01, upper_q=0.99)),
        ("pp", ct),
        ("model", LinearRegression()),
    ])
    return TransformedTargetRegressor(
        regressor=lr_pipe, func=np.log1p, inverse_func=np.expm1
    )


def build_lr_ordinal():
    ct = ColumnTransformer([
        ("sk", make_sk_pipe(True, True), SKEWED),
        ("num", make_reg_pipe(True), REGULAR_WITH_YEAR),
        ("ord", make_ordinal_q(), ORDINAL_Q),
        ("bsmt", make_bsmt(), ["BsmtQual"]),
        ("cat", make_ohe(), CAT_NOMINAL),
    ], remainder="drop")
    lr_pipe = Pipeline([
        ("domain", DomainImputer()),
        ("rare", RareCategoryGrouper(columns=CAT_NOMINAL, min_freq=0.01)),
        ("winsor", QuantileClipper(columns=CLIP_YEAR, lower_q=0.01, upper_q=0.99)),
        ("pp", ct),
        ("model", LinearRegression()),
    ])
    return TransformedTargetRegressor(
        regressor=lr_pipe, func=np.log1p, inverse_func=np.expm1
    )


def build_lr_final():
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ("pp", build_preprocessor(scale_numeric=True)),
            ("model", LinearRegression()),
        ]),
        func=np.log1p, inverse_func=np.expm1,
    )


def build_rf_baseline():
    ct = ColumnTransformer([
        ("num", make_reg_pipe(False), SKEWED + REGULAR_WITH_YEAR),
        ("cat", make_ohe(), CAT_OHE),
    ], remainder="drop")
    return Pipeline([
        ("domain", DomainImputer()),
        ("rare", RareCategoryGrouper(columns=CAT_OHE, min_freq=0.01)),
        ("winsor", QuantileClipper(columns=CLIP_YEAR, lower_q=0.01, upper_q=0.99)),
        ("pp", ct),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def build_rf_final():
    return Pipeline([
        ("pp", build_preprocessor(scale_numeric=False)),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])


def run_eval(label, model, X_tr, y_tr, X_te, y_te, kf):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    rmse = mean_squared_error(y_te, preds) ** 0.5
    r2 = r2_score(y_te, preds)
    cv = -cross_val_score(
        model, X_tr, y_tr, cv=kf,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    delta_mae = ""
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2,
            "cv_mean": cv.mean(), "cv_std": cv.std()}


def main():
    df = pd.read_csv("data/raw/train.csv")
    X, y = get_training_frame(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    configs_lr = [
        ("A) Baseline LR", build_lr_baseline),
        ("B) +log1p(SalePrice)", build_lr_with_logtarget),
        ("C) +log1p(features skew>1)", build_lr_log_features),
        ("D) +OrdinalEncoder calidad", build_lr_ordinal),
        ("E) +HouseAge/YearsSinceRemodel", build_lr_final),
    ]

    configs_rf = [
        ("F) Baseline RF", build_rf_baseline),
        ("G) RF Final (nuevas features)", build_rf_final),
    ]

    rows = []
    print("\n=== ABLACION LINEAL: acumulacion de mejoras ===")
    baseline_rmse = None
    for label, builder in configs_lr:
        r = run_eval(label, builder(), X_tr, y_tr, X_te, y_te, kf)
        rows.append(r)
        delta = f"  dRMSE={r['RMSE']-baseline_rmse:+,.0f}" if baseline_rmse else ""
        print(f"{label}: MAE={r['MAE']:,.0f}  RMSE={r['RMSE']:,.0f}  R2={r['R2']:.4f}"
              f"  cv={r['cv_mean']:,.0f}+/-{r['cv_std']:,.0f}{delta}")
        if baseline_rmse is None:
            baseline_rmse = r["RMSE"]

    print("\n=== RANDOM FOREST: efecto del nuevo preprocessor ===")
    baseline_rmse_rf = None
    for label, builder in configs_rf:
        r = run_eval(label, builder(), X_tr, y_tr, X_te, y_te, kf)
        rows.append(r)
        delta = f"  dRMSE={r['RMSE']-baseline_rmse_rf:+,.0f}" if baseline_rmse_rf else ""
        print(f"{label}: MAE={r['MAE']:,.0f}  RMSE={r['RMSE']:,.0f}  R2={r['R2']:.4f}"
              f"  cv={r['cv_mean']:,.0f}+/-{r['cv_std']:,.0f}{delta}")
        if baseline_rmse_rf is None:
            baseline_rmse_rf = r["RMSE"]

    print("\n=== RESUMEN COMPARATIVO ===")
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv("reports/ablation_study.csv", index=False)
    print(ablation_df[["label", "MAE", "RMSE", "R2", "cv_mean"]].to_string(index=False))
    print("\nGuardado en reports/ablation_study.csv")


if __name__ == "__main__":
    main()
