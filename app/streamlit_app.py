"""Streamlit app for interactive housing price prediction."""

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
from app.config import PRIMARY_INPUTS
from app.services import DataRepository, InputPreparationService
from app.ui import (
    FeatureInputRenderer,
    inject_styles,
    render_metrics_sidebar,
    render_result_card,
    render_workflow_cards,
)
from app.validation import InputValidator
from src.preprocessing import build_prediction_frame_from_dict, load_features


def render_header(features: list[str]) -> None:
    st.markdown(
        f"""
        <div class="hero-box">
            <h2>Prediccion de Precio de Viviendas</h2>
            <p class="muted" style="margin-bottom:0;">
                Interfaz ejecutiva para estimar precio, comparar modelos y explorar mejoras realistas de la vivienda.
            </p>
            <div class="hero-badges">
                <span class="hero-badge">{len(features)} variables activas</span>
                <span class="hero-badge">2 modelos en competencia</span>
                <span class="hero-badge">Vista esencial y detallada</span>
                <span class="hero-badge">Alineado al notebook final</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(features: list[str], metrics_df: pd.DataFrame | None) -> dict[str, object]:
    with st.sidebar:
        st.header("Panel de decision")
        st.caption("Ajusta el nivel de detalle y el modo de entrada sin sobrecargar la vista.")

        detail_level = st.radio(
            "Nivel de detalle",
            ["Esencial", "Detallado"],
            index=0,
            help="Esencial muestra solo lo clave. Detallado muestra analisis y contexto extra.",
        )
        is_detailed = detail_level == "Detallado"

        mode = st.radio(
            "Modo de entrada",
            [
                "Rapido (3 variables + defaults)",
                "Completo (editar todas las variables)",
            ],
            index=0,
        )

        show_advanced_inputs = False
        if mode.startswith("Rapido"):
            show_advanced_inputs = st.toggle(
                "Mostrar ajustes avanzados de entrada",
                value=False,
                help="Activa esta opcion solo si necesitas personalizar mas variables en esta prediccion.",
            )

        preset = st.selectbox(
            "Preset de vivienda",
            [
                "Base (mediana/moda)",
                "Casa compacta",
                "Casa familiar",
                "Casa premium",
            ],
            index=0,
        )

        st.caption(f"Variables activas: {len(features)}")
        if is_detailed:
            render_metrics_sidebar(metrics_df)
        else:
            st.caption("Vista esencial activa, se muestran solo controles y resultados clave.")

        st.subheader("Simulador A/B")
        compare_mode = st.toggle(
            "Activar simulador A/B",
            value=True,
            help="Compara el escenario actual contra una mejora realista de la vivienda.",
        )

        area_delta_pct = 0.0
        quality_delta = 0
        garage_delta = 0
        remodel_delta_years = 0

        if compare_mode:
            with st.expander("Configurar mejora", expanded=False):
                area_delta_pct = float(
                    st.slider(
                        f"Cambio en {FeatureCatalog.label('GrLivArea')} (%)",
                        min_value=-25,
                        max_value=40,
                        value=10,
                        step=1,
                    )
                )
                quality_delta = int(
                    st.slider(
                        f"Cambio en {FeatureCatalog.label('OverallQual')}",
                        min_value=-2,
                        max_value=2,
                        value=1,
                        step=1,
                    )
                )
                garage_delta = int(
                    st.slider(
                        f"Cambio en {FeatureCatalog.label('GarageCars')}",
                        min_value=-1,
                        max_value=2,
                        value=1,
                        step=1,
                    )
                )
                remodel_delta_years = int(
                    st.slider(
                        f"Incremento de {FeatureCatalog.label('YearRemodAdd')} (anios)",
                        min_value=0,
                        max_value=25,
                        value=5,
                        step=1,
                    )
                )

    return {
        "detail_level": detail_level,
        "mode": mode,
        "show_advanced_inputs": show_advanced_inputs,
        "preset": preset,
        "compare_mode": compare_mode,
        "area_delta_pct": area_delta_pct,
        "quality_delta": quality_delta,
        "garage_delta": garage_delta,
        "remodel_delta_years": remodel_delta_years,
    }


def render_glossary(features: list[str]) -> None:
    with st.expander("Guia rapida de variables (lenguaje simple)", expanded=False):
        for feature in features:
            help_text = FeatureCatalog.help_text(feature)
            if help_text:
                st.write(f"- {FeatureCatalog.label(feature)}: {help_text}")
            else:
                st.write(f"- {FeatureCatalog.label(feature)}")


def render_prediction_form(
    mode: str,
    features: list[str],
    groups: list[tuple[str, list[str]]],
    defaults: dict[str, object],
    renderer: FeatureInputRenderer,
    detailed_view: bool,
    show_advanced_inputs: bool,
) -> tuple[bool, dict[str, object]]:
    with st.form("predict_form", clear_on_submit=False):
        if mode.startswith("Rapido"):
            st.write("Configura 3 variables clave: area habitable, barrio y dormitorios.")
            quick_defaults = {k: defaults[k] for k in PRIMARY_INPUTS}
            quick_values = renderer.render_feature_grid(
                features=PRIMARY_INPUTS,
                defaults=quick_defaults,
                key_prefix="quick",
            )

            full_values = dict(defaults)
            full_values.update(quick_values)

            if show_advanced_inputs:
                with st.expander("Ajustes avanzados", expanded=detailed_view):
                    optional_features = [f for f in features if f not in PRIMARY_INPUTS]
                    optional_groups: list[tuple[str, list[str]]] = []

                    for title, group_features in groups:
                        extra = [feature for feature in group_features if feature in optional_features]
                        if extra:
                            optional_groups.append((title, extra))

                    optional_defaults = {feature: full_values[feature] for feature in optional_features}
                    optional_values = renderer.render_grouped_features(
                        groups=optional_groups,
                        defaults=optional_defaults,
                        key_prefix="optional",
                    )
                    full_values.update(optional_values)
            else:
                st.caption("Usa ajustes avanzados solo cuando necesites mayor precision en casos especiales.")
        else:
            st.write("Modo completo: puedes editar todo el vector de entrada.")
            full_values = renderer.render_grouped_features(
                groups=groups,
                defaults=defaults,
                key_prefix="full",
            )

        submitted = st.form_submit_button("Calcular precio")

    return submitted, full_values


def render_validation_notes(notes: list[str], title: str, detailed_view: bool) -> None:
    if not notes:
        return
    if not detailed_view:
        st.warning(f"{len(notes)} ajustes automaticos aplicados para mantener entradas validas.")
        return
    with st.expander(title, expanded=False):
        for note in notes:
            st.write(f"- {note}")


def main() -> None:
    st.set_page_config(page_title="Prediccion de Viviendas", page_icon="house", layout="wide")
    inject_styles()

    features = load_features()
    render_header(features)

    linear_model, rf_model = DataRepository.load_models()
    if linear_model is None or rf_model is None:
        st.error("No se encontraron modelos entrenados. Ejecuta: python -m src.train")
        st.stop()

    groups = InputPreparationService.resolve_feature_groups(features)
    ref_df = DataRepository.load_reference_data()
    metrics_df = DataRepository.load_metrics()

    sidebar_state = render_sidebar(features=features, metrics_df=metrics_df)
    is_detailed = str(sidebar_state["detail_level"]) == "Detallado"

    renderer = FeatureInputRenderer(ref_df=ref_df, show_rule_details=is_detailed)
    base_defaults = InputPreparationService.build_defaults(features, ref_df)
    defaults = InputPreparationService.apply_preset(
        preset=str(sidebar_state["preset"]),
        defaults=base_defaults,
        ref_df=ref_df,
    )

    tab_prediction, tab_context, tab_history = st.tabs(["Prediccion", "Contexto", "Historial"])

    with tab_prediction:
        st.markdown("<div class='section-title'>Entrada y estimacion</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Usa el modo rapido para decidir en segundos o el modo completo para ajustar todo el vector.</div>",
            unsafe_allow_html=True,
        )
        render_workflow_cards(
            [
                ("Paso 1", "Elige preset y ajusta los campos clave de la vivienda."),
                ("Paso 2", "Pulsa Calcular precio para obtener una estimacion inmediata."),
                ("Paso 3", "Si activas simulador A/B, compara el impacto de una mejora."),
            ]
        )

        if ref_df is None:
            st.warning("No se encontro train.csv, se usaran defaults genericos donde aplique.")
        else:
            st.caption("Defaults y presets calculados desde train.csv.")

        submitted, form_values = render_prediction_form(
            mode=str(sidebar_state["mode"]),
            features=features,
            groups=groups,
            defaults=defaults,
            renderer=renderer,
            detailed_view=is_detailed,
            show_advanced_inputs=bool(sidebar_state["show_advanced_inputs"]),
        )

        if not submitted:
            st.info("Completa los campos del formulario y presiona Predecir para ver la comparacion entre modelos.")
        else:
            sample_values = dict(defaults)
            sample_values.update(form_values)

            sanitized_values, validation_notes = InputValidator.sanitize_values(
                values=sample_values,
                features=features,
                ref_df=ref_df,
                defaults=defaults,
            )
            render_validation_notes(
                validation_notes,
                "Ajustes automaticos de calidad de datos",
                detailed_view=is_detailed,
            )

            sample = build_prediction_frame_from_dict(sanitized_values)
            pred_linear = float(linear_model.predict(sample)[0])
            pred_rf = float(rf_model.predict(sample)[0])
            pred_consensus = PredictionAnalytics.weighted_consensus(pred_linear, pred_rf, metrics_df)

            compare_mode = bool(sidebar_state["compare_mode"])
            compare_changes: list[str] = []
            pred_linear_b = pred_rf_b = pred_consensus_b = None

            if compare_mode:
                compare_values, compare_changes = InputPreparationService.build_ab_scenario(
                    base_values=sanitized_values,
                    ref_df=ref_df,
                    area_delta_pct=float(sidebar_state["area_delta_pct"]),
                    quality_delta=int(sidebar_state["quality_delta"]),
                    garage_delta=int(sidebar_state["garage_delta"]),
                    remodel_delta_years=int(sidebar_state["remodel_delta_years"]),
                )

                compare_values, compare_notes = InputValidator.sanitize_values(
                    values=compare_values,
                    features=features,
                    ref_df=ref_df,
                    defaults=defaults,
                )
                render_validation_notes(
                    compare_notes,
                    "Ajustes automaticos aplicados a escenario B",
                    detailed_view=is_detailed,
                )

                sample_b = build_prediction_frame_from_dict(compare_values)
                pred_linear_b = float(linear_model.predict(sample_b)[0])
                pred_rf_b = float(rf_model.predict(sample_b)[0])
                pred_consensus_b = PredictionAnalytics.weighted_consensus(pred_linear_b, pred_rf_b, metrics_df)

            level, level_text, diff_abs, diff_ratio = PredictionAnalytics.disagreement_level(
                pred_linear,
                pred_rf,
                metrics_df,
            )

            history = st.session_state.get("prediction_history", [])
            history.append(
                {
                    "Regresion Lineal": pred_linear,
                    "Random Forest": pred_rf,
                    "Consenso": pred_consensus,
                    "Diferencia": diff_abs,
                }
            )
            st.session_state["prediction_history"] = history[-15:]

            st.markdown("<div class='section-title'>Resultado</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='section-subtitle'>Nivel de discrepancia entre modelos: {level_text}</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns([1.45, 1])
            with c1:
                render_result_card(
                    "Precio recomendado",
                    f"${pred_consensus:,.2f}",
                    "Estimacion combinada de ambos modelos para una decision rapida.",
                )
            with c2:
                render_result_card(
                    "Diferencia entre modelos",
                    f"${diff_abs:,.2f}",
                    "Mientras mas baja sea, mas estable es la prediccion.",
                )

            if level == "baja":
                st.success(level_text)
            elif level == "media":
                st.warning(level_text)
            else:
                st.error(level_text)

            if is_detailed:
                with st.expander("Ver detalle por modelo", expanded=False):
                    c_model_1, c_model_2 = st.columns(2)
                    with c_model_1:
                        render_result_card("Regresion Lineal", f"${pred_linear:,.2f}", "Estimacion del modelo lineal.")
                    with c_model_2:
                        render_result_card("Random Forest", f"${pred_rf:,.2f}", "Estimacion del modelo no lineal.")

                    st.caption(f"Cobertura de features en inferencia: {int(sample.notna().sum(axis=1).iloc[0])}/{len(features)}")
                    st.progress(min(diff_ratio / 0.60, 1.0))
                    st.caption("Indice visual de discrepancia entre modelos.")

                    chart_df = pd.DataFrame(
                        {
                            "Modelo": ["Regresion Lineal", "Random Forest", "Consenso"],
                            "Prediccion": [pred_linear, pred_rf, pred_consensus],
                        }
                    ).set_index("Modelo")
                    st.bar_chart(chart_df)

            if compare_mode and pred_linear_b is not None and pred_rf_b is not None and pred_consensus_b is not None:
                st.markdown("### Comparacion A/B orientada a decision")

                delta_consensus = pred_consensus_b - pred_consensus
                a_col, b_col, d_col = st.columns(3)
                a_col.metric("Escenario A (actual)", f"${pred_consensus:,.2f}")
                b_col.metric("Escenario B (simulado)", f"${pred_consensus_b:,.2f}")
                d_col.metric("Impacto estimado", f"${delta_consensus:,.2f}")

                if compare_changes and is_detailed:
                    st.caption("Cambios aplicados en escenario B:")
                    for change in compare_changes:
                        st.write(f"- {change}")

                if is_detailed:
                    compare_df = pd.DataFrame(
                        {
                            "Escenario A": [pred_linear, pred_rf, pred_consensus],
                            "Escenario B": [pred_linear_b, pred_rf_b, pred_consensus_b],
                        },
                        index=["Regresion Lineal", "Random Forest", "Consenso"],
                    )
                    st.bar_chart(compare_df)

                if pred_consensus != 0:
                    pct = (delta_consensus / pred_consensus) * 100.0
                    st.info(f"Cambio relativo estimado del consenso: {pct:+.2f}%")

            if is_detailed:
                with st.expander("Diagnostico", expanded=False):
                    st.write("- Regresion Lineal captura tendencia global, Random Forest captura patrones no lineales por segmentos.")
                    st.write("- Diferencias altas aparecen cuando la combinacion de features es rara o esta en los bordes del dataset.")
                    st.write(
                        "- Si la diferencia sube mucho, revisa primero Barrio o zona, Calidad general de materiales y acabados, "
                        "Area habitable sobre nivel del suelo y Area de garage."
                    )

                    extreme_notes = PredictionAnalytics.detect_extreme_inputs(sample, ref_df)
                    if extreme_notes:
                        st.caption("Variables en zonas extremas del dataset:")
                        for note in extreme_notes:
                            st.write(f"- {note}")

                    if ref_df is not None and "SalePrice" in ref_df.columns:
                        p_linear = PredictionAnalytics.percentile_rank(ref_df["SalePrice"], pred_linear)
                        p_rf = PredictionAnalytics.percentile_rank(ref_df["SalePrice"], pred_rf)
                        p_cons = PredictionAnalytics.percentile_rank(ref_df["SalePrice"], pred_consensus)

                        st.caption("Contexto percentil en distribucion historica:")
                        if p_linear is not None:
                            st.write(f"- Regresion Lineal: {p_linear:.1f}%")
                        if p_rf is not None:
                            st.write(f"- Random Forest: {p_rf:.1f}%")
                        if p_cons is not None:
                            st.write(f"- Consenso: {p_cons:.1f}%")

    with tab_context:
        st.markdown("<div class='section-title'>Contexto del sistema</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>La app usa 22 variables, dos modelos y reglas de negocio para calidad y coherencia de entrada.</div>",
            unsafe_allow_html=True,
        )

        if metrics_df is not None and not metrics_df.empty:
            best_row = metrics_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                render_result_card("Modelo lider", str(best_row.get("Model", "-")), "Mejor RMSE historico dentro de la app.")
            with c2:
                render_result_card("RMSE lider", f"{float(best_row.get('RMSE', 0.0)):,.2f}", "Menor error absoluto cuadratico medio.")
            with c3:
                render_result_card("R2 lider", f"{float(best_row.get('R2', 0.0)):.4f}", "Mayor capacidad explicativa.")

        st.markdown("#### Mapa de variables")
        st.write("La interfaz prioriza estas familias de atributos:")
        for title, group_features in groups:
            st.write(f"- {title}: {', '.join(FeatureCatalog.label(feature) for feature in group_features)}")

        if ref_df is None:
            st.warning("No se encontro train.csv, por lo que el contexto estadistico es limitado.")
        elif is_detailed:
            render_glossary(features)

        if metrics_df is not None and not metrics_df.empty:
            st.markdown("#### Tabla de metricas historicas")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with tab_history:
        st.markdown("<div class='section-title'>Historial de corridas</div>", unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.get("prediction_history", []))
        if history_df.empty:
            st.info("Todavia no hay predicciones guardadas en esta sesion.")
        else:
            history_df = history_df.reset_index(drop=True)
            history_df.index = history_df.index + 1
            history_df.index.name = "Corrida"
            st.line_chart(history_df[["Regresion Lineal", "Random Forest", "Consenso"]])
            st.dataframe(history_df, use_container_width=True)


if __name__ == "__main__":
    main()
