"""Streamlit app for interactive housing price prediction (premium dark UI)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analytics import PredictionAnalytics
from app.catalog import FeatureCatalog
from app.config import CATEGORICAL_OPTION_LABELS_BY_FEATURE, INTEGER_LIKE_FEATURES
from app.services import DataRepository, InputPreparationService
from app.ui import (
    inject_styles,
    render_bars,
    render_empty_state,
    render_header,
    render_hero,
    render_section,
    render_table,
)
from app.validation import InputValidator
from src.preprocessing import build_prediction_frame_from_dict, load_features

MODEL_LABELS = {"linear_regression": "Regresión Lineal", "random_forest": "Random Forest"}

# Feature typing for the right widget per variable.
SCALE_1_10 = {"OverallQual", "OverallCond"}
YEAR_FEATURES = {"YearBuilt", "YearRemodAdd"}
AREA_FEATURES = {"GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LotArea", "GarageArea"}


def _bounds(ref_df: pd.DataFrame | None, feature: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if ref_df is not None and feature in ref_df.columns:
        low, high = InputPreparationService.numeric_bounds(ref_df[feature])
        return float(low), float(high)
    return fallback


def _quality_options(feature: str, ref_df: pd.DataFrame | None) -> list[str]:
    """Real quality levels for a categorical feature, ordered best -> worst."""
    label_map = CATEGORICAL_OPTION_LABELS_BY_FEATURE.get(feature, {})
    data_opts: list[str] = []
    if ref_df is not None and feature in ref_df.columns:
        data_opts = sorted(ref_df[feature].dropna().astype(str).unique().tolist())

    if label_map:
        ordered = [c for c in label_map if c != "Missing" and (not data_opts or c in data_opts)]
        ordered += [c for c in data_opts if c not in ordered and c not in label_map]
        return ordered or [c for c in label_map if c != "Missing"]
    return data_opts or ["Missing"]


def render_feature(feature: str, ref_df: pd.DataFrame | None, default: object) -> object:
    label = FeatureCatalog.label(feature)
    help_text = FeatureCatalog.help_text(feature)
    key = f"in_{feature}"

    # 1-10 quality / condition scales
    if feature in SCALE_1_10:
        value = int(round(InputPreparationService.to_float(default, 5.0)))
        return st.slider(label, 1, 10, max(1, min(value, 10)), help=help_text or "1 = muy basica · 10 = excelente")

    # Categorical: friendly, real-world levels
    is_categorical = (ref_df is not None and feature in ref_df.columns
                      and not pd.api.types.is_numeric_dtype(ref_df[feature]))
    if is_categorical or feature in CATEGORICAL_OPTION_LABELS_BY_FEATURE:
        options = _quality_options(feature, ref_df)
        label_map = CATEGORICAL_OPTION_LABELS_BY_FEATURE.get(feature, {})
        default_code = str(default) if str(default) in options else options[0]
        return st.selectbox(
            label, options, index=options.index(default_code),
            format_func=lambda c: label_map.get(c, c), help=help_text, key=key,
        )

    # Numeric
    low, high = _bounds(ref_df, feature, (0.0, 5000.0))
    value = InputPreparationService.to_float(default, (low + high) / 2)
    value = max(low, min(value, high))

    if feature in YEAR_FEATURES:
        return st.slider(label, int(low), int(high), int(value), step=1, help=help_text, key=key)
    if feature in AREA_FEATURES:
        step = max(10, int((high - low) / 100))
        return st.slider(label, int(low), int(high), int(value), step=step, help=help_text, key=key)
    if feature in INTEGER_LIKE_FEATURES:
        return st.number_input(label, int(low), int(high), int(round(value)), step=1, help=help_text, key=key)
    return st.number_input(label, float(low), float(high), float(value), step=1.0, help=help_text, key=key)


def render_form(features: list[str], defaults: dict[str, object], ref_df: pd.DataFrame | None) -> tuple[bool, dict[str, object]]:
    groups = InputPreparationService.resolve_feature_groups(features)
    render_section(
        "Parámetros de la vivienda",
        "El modelo usa las 22 variables para la estimación. Ajusta las que conozcas; el resto toma valores típicos.",
    )
    values: dict[str, object] = {}
    with st.form("predict_form"):
        tabs = st.tabs([title for title, _ in groups])
        for tab, (_, group_features) in zip(tabs, groups):
            with tab:
                cols = st.columns(2)
                for idx, feature in enumerate(group_features):
                    with cols[idx % 2]:
                        values[feature] = render_feature(feature, ref_df, defaults.get(feature))
        submitted = st.form_submit_button("Calcular precio")
    return submitted, values


def compute_result(
    form_values: dict[str, object],
    features: list[str],
    defaults: dict[str, object],
    ref_df: pd.DataFrame | None,
    metrics_df: pd.DataFrame | None,
    linear_model,
    rf_model,
) -> dict[str, object]:
    sample_values = dict(defaults)
    sample_values.update(form_values)
    sanitized, notes = InputValidator.sanitize_values(sample_values, features, ref_df, defaults)
    sample = build_prediction_frame_from_dict(sanitized)

    pred_linear = float(linear_model.predict(sample)[0])
    pred_rf = float(rf_model.predict(sample)[0])
    consensus = PredictionAnalytics.weighted_consensus(pred_linear, pred_rf, metrics_df)

    leader_key = "random_forest"
    if metrics_df is not None and not metrics_df.empty and "Model" in metrics_df.columns:
        leader_key = str(metrics_df.iloc[0]["Model"])
    low, high = PredictionAnalytics.price_range(consensus, metrics_df, leader_key)

    percentile = None
    if ref_df is not None and "SalePrice" in ref_df.columns:
        percentile = PredictionAnalytics.percentile_rank(ref_df["SalePrice"], consensus)

    return {
        "pred_linear": pred_linear, "pred_rf": pred_rf, "consensus": consensus,
        "low": low, "high": high, "leader": MODEL_LABELS.get(leader_key, leader_key),
        "notes": notes, "percentile": percentile,
    }


def render_result(result: dict[str, object], metrics_df: pd.DataFrame | None, rf_model) -> None:
    render_hero(
        float(result["consensus"]), float(result["low"]), float(result["high"]),
        f"Estimación combinada · modelo líder: {result['leader']}",
    )

    notes = result.get("notes") or []
    if notes:
        with st.expander(f"Se ajustaron {len(notes)} entrada(s) para mantener datos válidos"):
            for note in notes:
                st.write(f"- {note}")

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Regresión Lineal", f"${result['pred_linear']:,.0f}")
    c2.metric("Random Forest", f"${result['pred_rf']:,.0f}")
    c3.metric("Diferencia entre modelos", f"${abs(result['pred_rf'] - result['pred_linear']):,.0f}")

    st.write("")
    tab_compare, tab_drivers, tab_perf = st.tabs(
        ["Comparación de modelos", "Variables más influyentes", "Rendimiento"]
    )

    with tab_compare:
        render_section("Qué predice cada modelo", "La estimación final combina ambos, ponderando por su precisión histórica.")
        preds = [
            ("Regresión Lineal", float(result["pred_linear"]), False),
            ("Random Forest", float(result["pred_rf"]), False),
            ("Estimación combinada", float(result["consensus"]), True),
        ]
        mx = max(v for _, v, _ in preds) or 1.0
        render_bars([{"label": l, "value": f"${v:,.0f}", "frac": v / mx, "strong": s} for l, v, s in preds])
        if result.get("percentile") is not None:
            st.caption(f"Esta vivienda quedaría por encima del {float(result['percentile']):.0f}% de las casas del histórico.")

    with tab_drivers:
        render_section("Qué pesa más en el precio", "Importancia de variables aprendida por el Random Forest.")
        importance_df = PredictionAnalytics.feature_importances(rf_model, top_n=8)
        if importance_df is None or importance_df.empty:
            st.info("El modelo no expone importancia de variables.")
        else:
            mx = float(importance_df["Importancia"].max()) or 1.0
            render_bars([
                {"label": row.Variable, "value": f"{row.Importancia * 100:.0f}%",
                 "frac": row.Importancia / mx, "strong": i == 0}
                for i, row in enumerate(importance_df.itertuples())
            ])
            top = importance_df.iloc[0]
            st.caption(f"La variable más determinante es **{top['Variable']}** ({top['Importancia'] * 100:.0f}% del peso).")

    with tab_perf:
        render_section("Precisión de los modelos", "Métricas de error en datos no vistos (menor RMSE = mejor).")
        if metrics_df is None or metrics_df.empty:
            st.info("No se encontró reports/metrics_comparison.csv. Ejecuta: python -m src.train")
        else:
            rows = []
            for i, (_, r) in enumerate(metrics_df.iterrows()):
                name = MODEL_LABELS.get(str(r.get("Model", "")), str(r.get("Model", "")))
                if i == 0:
                    name += "<span class='badge'>Mejor</span>"
                rows.append([
                    name,
                    f"{float(r['MAE']):,.0f}" if "MAE" in r else "-",
                    f"{float(r['MSE']):,.0f}" if "MSE" in r else "-",
                    f"{float(r['RMSE']):,.0f}" if "RMSE" in r else "-",
                    f"{float(r['R2']):.4f}" if "R2" in r else "-",
                ])
            render_table(["Modelo", "MAE", "MSE", "RMSE", "R²"], rows, best_row=0)


def main() -> None:
    st.set_page_config(page_title="Estimador de Precio de Viviendas", page_icon="🏠", layout="wide")

    mode = st.session_state.get("theme_mode", "dark")
    top_left, top_right = st.columns([5, 1])
    with top_right:
        light = st.toggle("Modo claro", value=(mode == "light"), key="light_toggle")
    mode = "light" if light else "dark"
    st.session_state["theme_mode"] = mode
    inject_styles(mode)

    with top_left:
        render_header(
            "Estimador de Precio de Viviendas",
            "Predice el precio de una vivienda y compara dos modelos de machine learning en tiempo real.",
        )
    st.write("")

    linear_model, rf_model = DataRepository.load_models()
    if linear_model is None or rf_model is None:
        st.error("No se encontraron modelos entrenados. Ejecuta primero: `python -m src.train`")
        st.stop()

    features = load_features()
    ref_df = DataRepository.load_reference_data()
    metrics_df = DataRepository.load_metrics()
    defaults = InputPreparationService.build_defaults(features, ref_df)

    submitted, form_values = render_form(features, defaults, ref_df)
    if submitted:
        st.session_state["result"] = compute_result(
            form_values, features, defaults, ref_df, metrics_df, linear_model, rf_model
        )

    st.write("")
    result = st.session_state.get("result")
    if result is None:
        render_empty_state()
    else:
        render_result(result, metrics_df, rf_model)


if __name__ == "__main__":
    main()
