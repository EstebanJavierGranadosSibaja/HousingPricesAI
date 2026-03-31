"""Predict housing prices with trained models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from preprocessing import build_prediction_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_model(model_name: str, models_dir: Path):
    model_path = models_dir / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro {model_path}. Ejecuta primero: python src/train.py"
        )
    return joblib.load(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predecir precio de vivienda")
    parser.add_argument("--tamano", type=float, required=True, help="Tamano en m2")
    parser.add_argument("--ubicacion", type=str, required=True, help="Barrio o zona")
    parser.add_argument("--habitaciones", type=int, required=True, help="Numero de habitaciones")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directorio de modelos entrenados",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    linear_model = load_model("linear_regression", args.models_dir)
    rf_model = load_model("random_forest", args.models_dir)

    sample = build_prediction_frame(args.tamano, args.ubicacion, args.habitaciones)

    linear_pred = float(linear_model.predict(sample)[0])
    rf_pred = float(rf_model.predict(sample)[0])

    print("\nPrediccion de precio")
    print(f"Regresion Lineal: {linear_pred:,.2f}")
    print(f"Random Forest:    {rf_pred:,.2f}")
    print(f"Diferencia abs:   {abs(rf_pred - linear_pred):,.2f}")


if __name__ == "__main__":
    main()
