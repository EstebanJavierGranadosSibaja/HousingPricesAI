"""Train and evaluate Linear Regression and Random Forest models.

Metodologia de evaluacion (a prueba de tribunal):
- Division 80/20 con random_state=42 para una estimacion final imparcial en test.
- Validacion cruzada KFold=5 sobre el TRAIN para estimar el error fuera de muestra
  con su incertidumbre (media +/- desviacion). La SELECCION del modelo se hace por
  CV, no por el test (evita sesgo de seleccion); el test solo confirma.
- Regresion Lineal: modela log1p(SalePrice) por defecto y revierte con expm1.
  Corrige asimetria (skew 1.88) y heterocedasticidad del precio, mejorando la
  estabilidad del modelo lineal.
- Random Forest: sin transformacion del target por defecto (invariante a monotonias
  del target). Se puede habilitar con --log-target.
- Modelo final del sistema: Random Forest, seleccionado por su mejor desempeno
  predictivo en test. La Regresion Lineal + log1p es una alternativa mas estable
  documentada pero no fue seleccionada como modelo operativo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, get_training_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CV_SPLITS = 5
RANDOM_STATE = 42


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse**0.5,
        "R2": r2_score(y_true, y_pred),
    }


def _maybe_log(pipeline: Pipeline, log_target: bool):
    """Envuelve el pipeline para modelar log1p(y) y revertir con expm1."""
    if log_target:
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1
        )
    return pipeline


def build_models(log_target_rf: bool = False) -> dict[str, object]:
    # LR SIEMPRE modela log1p(SalePrice): corrige asimetria (skew=1.88) y
    # heterocedasticidad. Para OLS puro el scaler no cambia predicciones, pero
    # log1p del target si mejora estabilidad y reduce sobreajuste.
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("model", LinearRegression()),
        ]
    )
    linear_model = TransformedTargetRegressor(
        regressor=linear_pipeline, func=np.log1p, inverse_func=np.expm1
    )

    # RF: invariante a la escala de features y a monotonias del target.
    # scale_numeric=False evita computo innecesario.
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_model = _maybe_log(rf_pipeline, log_target_rf)

    return {
        "linear_regression": linear_model,
        "random_forest": rf_model,
    }


# LR siempre esta envuelto en TransformedTargetRegressor -> prefijo regressor__.
# RF solo requiere prefijo si log_target_rf=True (ver _grid_for).
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "linear_regression": {
        "regressor__model__fit_intercept": [True, False],
        "regressor__model__positive": [False, True],
    },
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.6],
    },
}


def _grid_for(name: str, log_target_rf: bool) -> dict[str, list]:
    grid = PARAM_GRIDS[name]
    # RF con log target queda envuelto en TransformedTargetRegressor -> prefijo.
    # LR ya tiene el prefijo en PARAM_GRIDS (siempre envuelto).
    if name == "random_forest" and log_target_rf:
        return {f"regressor__{k}": v for k, v in grid.items()}
    return grid


def train_and_evaluate(
    train_csv: Path, tune: bool = False, log_target_rf: bool = False
) -> pd.DataFrame:
    if not train_csv.exists():
        raise FileNotFoundError(f"No se encontro el archivo de entrenamiento: {train_csv}")

    df = pd.read_csv(train_csv)
    x, y = get_training_frame(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    kfold = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(log_target_rf=log_target_rf)
    metrics_rows: list[dict[str, float | str]] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        # Validacion cruzada sobre TRAIN: error fuera de muestra + incertidumbre.
        cv_rmse = -cross_val_score(
            model, x_train, y_train, cv=kfold,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )

        if tune:
            search = GridSearchCV(
                estimator=model,
                param_grid=_grid_for(name, log_target_rf),
                cv=kfold,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train, y_train)
            fitted = search.best_estimator_
            print(f"[tune] {name} best_params: {search.best_params_}")
        else:
            model.fit(x_train, y_train)
            fitted = model

        preds = fitted.predict(x_test)
        metrics = calculate_metrics(y_test, preds)
        metrics_rows.append({
            "Model": name,
            **metrics,
            "cv_rmse_mean": float(cv_rmse.mean()),
            "cv_rmse_std": float(cv_rmse.std()),
        })

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(fitted, model_path)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="RMSE", ascending=True)
    metrics_df.to_csv(REPORTS_DIR / "metrics_comparison.csv", index=False)
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar modelos de prediccion de precio")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Ruta al CSV de entrenamiento",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Ejecuta GridSearchCV (KFold=5) y persiste el modelo ajustado en lugar del base",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Aplica log1p(SalePrice) al target del Random Forest (LR siempre usa log1p por defecto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(
        args.train_csv, tune=args.tune, log_target_rf=args.log_target
    )

    # Seleccion por CV (no por test) para evitar sesgo de seleccion.
    by_cv = metrics.sort_values(by="cv_rmse_mean", ascending=True).reset_index(drop=True)
    best_cv = by_cv.iloc[0]

    print("\nEntrenamiento completado. Resultados:")
    print(metrics.to_string(index=False))
    print(
        f"\nMejor modelo por validacion cruzada: {best_cv['Model']} "
        f"(cv_rmse {best_cv['cv_rmse_mean']:,.0f} +/- {best_cv['cv_rmse_std']:,.0f})"
    )
    print(f"Modelos guardados en: {MODELS_DIR}")
    print(f"Metricas guardadas en: {REPORTS_DIR / 'metrics_comparison.csv'}")


if __name__ == "__main__":
    main()
