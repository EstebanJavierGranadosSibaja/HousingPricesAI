"""Train and evaluate Linear Regression and Random Forest models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, get_training_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": mse**0.5,
        "R2": r2_score(y_true, y_pred),
    }


def build_models() -> dict[str, Pipeline]:
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", LinearRegression()),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "linear_regression": linear_pipeline,
        "random_forest": rf_pipeline,
    }


def train_and_evaluate(train_csv: Path) -> pd.DataFrame:
    if not train_csv.exists():
        raise FileNotFoundError(f"No se encontro el archivo de entrenamiento: {train_csv}")

    df = pd.read_csv(train_csv)
    x, y = get_training_frame(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    models = build_models()
    metrics_rows: list[dict[str, float | str]] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        metrics = calculate_metrics(y_test, preds)
        metrics_rows.append({"Model": name, **metrics})

        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_evaluate(args.train_csv)

    print("\nEntrenamiento completado. Resultados:")
    print(metrics.to_string(index=False))
    print(f"\nModelos guardados en: {MODELS_DIR}")
    print(f"Metricas guardadas en: {REPORTS_DIR / 'metrics_comparison.csv'}")


if __name__ == "__main__":
    main()
