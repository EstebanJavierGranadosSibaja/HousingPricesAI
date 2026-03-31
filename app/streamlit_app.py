"""Simple Streamlit app for interactive housing price prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import build_prediction_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


@st.cache_resource
def load_models():
    linear_path = MODELS_DIR / "linear_regression.joblib"
    rf_path = MODELS_DIR / "random_forest.joblib"

    if not linear_path.exists() or not rf_path.exists():
        return None, None

    return joblib.load(linear_path), joblib.load(rf_path)


@st.cache_data
def get_locations() -> list[str]:
    if not DATA_PATH.exists():
        return ["NAmes"]

    data = pd.read_csv(DATA_PATH)
    if "Neighborhood" not in data.columns:
        return ["NAmes"]

    locations = sorted(data["Neighborhood"].dropna().unique().tolist())
    return locations or ["NAmes"]


def main() -> None:
    st.set_page_config(page_title="Prediccion de Viviendas", page_icon="🏠", layout="centered")
    st.title("Prediccion de Precio de Viviendas")
    st.caption("Comparacion entre Regresion Lineal y Random Forest")

    linear_model, rf_model = load_models()

    if linear_model is None or rf_model is None:
        st.error("No se encontraron modelos entrenados. Ejecuta: python src/train.py")
        st.stop()

    locations = get_locations()

    tamano = st.number_input("Tamano (m2)", min_value=20.0, max_value=800.0, value=120.0, step=1.0)
    ubicacion = st.selectbox("Ubicacion", options=locations, index=0)
    habitaciones = st.number_input("Habitaciones", min_value=1, max_value=10, value=3, step=1)

    if st.button("Predecir"):
        sample = build_prediction_frame(tamano, ubicacion, habitaciones)
        pred_linear = float(linear_model.predict(sample)[0])
        pred_rf = float(rf_model.predict(sample)[0])

        col1, col2 = st.columns(2)
        col1.metric("Regresion Lineal", f"${pred_linear:,.2f}")
        col2.metric("Random Forest", f"${pred_rf:,.2f}")

        diff = abs(pred_rf - pred_linear)
        st.info(f"Diferencia absoluta entre modelos: ${diff:,.2f}")


if __name__ == "__main__":
    main()
