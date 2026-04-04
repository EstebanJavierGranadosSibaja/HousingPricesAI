"""Predict housing prices with trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import build_prediction_frame_from_dict, load_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def load_model(model_name: str, models_dir: Path):
    model_path = models_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro {model_path}. Ejecuta primero: python -m src.train"
        )
    return joblib.load(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predecir precio de vivienda")
    parser.add_argument("--tamano", type=float, required=True, help="Tamano en m2")
    parser.add_argument("--ubicacion", type=str, required=True, help="Barrio o zona")
    parser.add_argument("--habitaciones", type=int, required=True, help="Numero de habitaciones")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="CSV de referencia para completar defaults de todas las features",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directorio de modelos entrenados",
    )
    return parser.parse_args()


def build_feature_defaults(train_csv: Path, features: list[str]) -> dict[str, object]:
    defaults: dict[str, object] = {}

    if not train_csv.exists():
        raise FileNotFoundError(
            f"No se encontro {train_csv}. Este script requiere train.csv para completar todas las features."
        )

    df = pd.read_csv(train_csv)

    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "train.csv no contiene todas las columnas del manifest: "
            + ", ".join(missing_features)
        )

    for feature in features:
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                defaults[feature] = float(numeric.median())
            else:
                defaults[feature] = 0.0
        else:
            mode_vals = series.dropna().astype(str).mode()
            if not mode_vals.empty:
                defaults[feature] = mode_vals.iloc[0]
            else:
                defaults[feature] = "Missing"

    return defaults


def main() -> None:
    args = parse_args()

    linear_model = load_model("linear_regression", args.models_dir)
    rf_model = load_model("random_forest", args.models_dir)

    features = load_features()
    defaults = build_feature_defaults(args.train_csv, features)

    user_values = {
        "GrLivArea": args.tamano,
        "Neighborhood": args.ubicacion,
        "BedroomAbvGr": args.habitaciones,
    }

    sample_values = {feature: defaults[feature] for feature in features}
    sample_values.update(user_values)
    sample = build_prediction_frame_from_dict(sample_values)

    linear_pred = float(linear_model.predict(sample)[0])
    rf_pred = float(rf_model.predict(sample)[0])

    print("\nPrediccion de precio")
    print(f"Features usadas: {sample.shape[1]} / {len(features)} (segun data/features.json)")
    print(f"Defaults tomados de: {args.train_csv}")
    print("Overrides del usuario:", ", ".join(user_values.keys()))
    print(f"Regresion Lineal: {linear_pred:,.2f}")
    print(f"Random Forest:    {rf_pred:,.2f}")
    print(f"Diferencia abs:   {abs(rf_pred - linear_pred):,.2f}")


if __name__ == "__main__":
    main()
