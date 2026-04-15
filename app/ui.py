"""UI helpers and widgets for Streamlit input rendering."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app.catalog import FeatureCatalog
from app.config import (
    INTEGER_LIKE_FEATURES,
    NUMERIC_FEATURE_HINTS,
    OVERALL_CONDITION_CHOICES,
    QUALITY_CATEGORY_RULESETS,
    QUALITY_CONDITION_FACTS,
    QUALITY_FINISH_FACTS,
    QUALITY_MATERIAL_FACTS,
)
from app.quality_expert import QualityExpertEngine
from app.services import InputPreparationService


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b1020;
            --sidebar: #0f172a;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-strong: #111827;
            --line: rgba(148, 163, 184, 0.18);
            --ink: #f8fafc;
            --muted: #cbd5e1;
            --accent: #f59e0b;
            --accent-2: #38bdf8;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(245, 158, 11, 0.14), transparent 30%),
                linear-gradient(180deg, #07111f 0%, #0b1020 38%, #0f172a 100%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid var(--line);
        }
        .main .block-container {
            padding-top: 0.7rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .hero-box {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.82));
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.22);
        }
        .hero-box h2 {
            color: var(--ink);
            letter-spacing: 0.2px;
            margin: 0;
            font-size: 1.95rem;
        }
        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }
        .hero-badge {
            border: 1px solid rgba(245, 158, 11, 0.25);
            background: rgba(245, 158, 11, 0.1);
            color: #fde68a;
            border-radius: 999px;
            padding: 0.3rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .section-title {
            margin: 0.4rem 0 0.25rem 0;
            color: var(--ink);
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .section-subtitle {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.5rem;
        }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 0.55rem;
            margin: 0.2rem 0 0.9rem 0;
        }
        .workflow-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.78));
            padding: 0.65rem 0.78rem;
        }
        .workflow-card .kicker {
            color: #f8fafc;
            font-size: 0.79rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .workflow-card .text {
            color: #cbd5e1;
            font-size: 0.78rem;
            line-height: 1.35;
        }
        .muted {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }
        .model-pill {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(15, 23, 42, 0.82);
            padding: 0.5rem 0.6rem;
            margin-bottom: 0.45rem;
        }
        .model-pill .title {
            color: #f8fafc;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.12rem;
        }
        .model-pill .meta {
            color: #cbd5e1;
            font-size: 0.74rem;
            line-height: 1.2;
        }
        .result-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(15, 23, 42, 0.92));
            padding: 0.9rem 1rem;
            box-shadow: 0 16px 36px rgba(2, 6, 23, 0.2);
        }
        .result-card .label {
            color: #cbd5e1;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }
        .result-card .value {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.18rem;
        }
        .result-card .caption {
            color: #94a3b8;
            font-size: 0.8rem;
            line-height: 1.3;
        }
        [data-testid="stForm"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.9rem 1rem 0.8rem 1rem;
        }
        [data-testid="stWidgetLabel"] p {
            color: #e2e8f0;
            font-size: 0.9rem;
            font-weight: 600;
        }
        [data-baseweb="input"] input,
        [data-baseweb="select"] div {
            background: rgba(30, 41, 59, 0.68);
            color: #f8fafc;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-size: 0.88rem;
            font-weight: 600;
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(56, 189, 248, 0.42);
            background: linear-gradient(90deg, rgba(14, 116, 144, 0.92), rgba(2, 132, 199, 0.92));
            color: #f8fafc;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        @media (max-width: 860px) {
            .hero-box h2 {
                font-size: 1.45rem;
            }
            .hero-badge {
                font-size: 0.73rem;
            }
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.66);
        }
        [data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.5rem 0.6rem;
            background: rgba(15, 23, 42, 0.78);
        }
        [data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }
        [data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        [data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, #38bdf8, #f59e0b);
        }
        .stAlert {
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">{title}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_cards(cards: list[tuple[str, str]]) -> None:
    card_blocks = "".join(
        [
            f"<div class='workflow-card'><div class='kicker'>{title}</div><div class='text'>{description}</div></div>"
            for title, description in cards
        ]
    )
    st.markdown(f"<div class='workflow-grid'>{card_blocks}</div>", unsafe_allow_html=True)


def render_metrics_sidebar(metrics_df: pd.DataFrame | None) -> None:
    if metrics_df is None or metrics_df.empty:
        return

    st.subheader("Rendimiento historico")
    best_row = metrics_df.iloc[0]
    st.caption(
        "Mejor modelo actual: "
        f"{str(best_row.get('Model', 'modelo'))}"
        f" | RMSE {float(best_row.get('RMSE', 0.0)):,.0f}"
    )

    with st.expander("Ver detalle de metricas", expanded=False):
        for _, row in metrics_df.iterrows():
            model = str(row.get("Model", "modelo"))
            rmse = row.get("RMSE", None)
            mae = row.get("MAE", None)
            r2 = row.get("R2", None)

            meta_chunks: list[str] = []
            if pd.notna(rmse):
                meta_chunks.append(f"RMSE: {float(rmse):,.0f}")
            if pd.notna(mae):
                meta_chunks.append(f"MAE: {float(mae):,.0f}")
            if pd.notna(r2):
                meta_chunks.append(f"R2: {float(r2):.3f}")

            st.markdown(
                f"""
                <div class="model-pill">
                    <div class="title">{model}</div>
                    <div class="meta">{' | '.join(meta_chunks)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


class FeatureInputRenderer:
    """Renders user-friendly feature widgets and expert systems."""

    def __init__(self, ref_df: pd.DataFrame | None, show_rule_details: bool = False):
        self.ref_df = ref_df
        self.show_rule_details = show_rule_details

    @staticmethod
    def _nearest_choice_index(choices: list[tuple[str, int]], target: int) -> int:
        if not choices:
            return 0
        return min(range(len(choices)), key=lambda idx: abs(choices[idx][1] - target))

    def render_overall_quality_expert_widget(
        self,
        feature: str,
        default_value: object,
        key_prefix: str,
    ) -> int:
        label = FeatureCatalog.label(feature)
        help_text = FeatureCatalog.help_text(feature)

        default_score = int(round(InputPreparationService.to_float(default_value, 5.0)))
        material_default, finish_default, condition_default = QualityExpertEngine.default_overall_profile(default_score)

        material_options = list(QUALITY_MATERIAL_FACTS.keys())
        finish_options = list(QUALITY_FINISH_FACTS.keys())
        condition_options = list(QUALITY_CONDITION_FACTS.keys())

        st.caption(f"{label}: seleccion asistida por reglas de negocio.")

        material_choice = st.selectbox(
            "1) Material predominante de acabados",
            options=material_options,
            index=material_options.index(material_default),
            key=f"{key_prefix}_overallqual_material",
            help="Ejemplo: madera, piedra, metal, marmol.",
        )

        finish_choice = st.selectbox(
            "2) Nivel de acabados",
            options=finish_options,
            index=finish_options.index(finish_default),
            key=f"{key_prefix}_overallqual_finish",
        )

        condition_choice = st.selectbox(
            "3) Estado de conservacion",
            options=condition_options,
            index=condition_options.index(condition_default),
            key=f"{key_prefix}_overallqual_condition",
            help=help_text,
        )

        inferred_score, rules = QualityExpertEngine.infer_overall_quality_score(
            material_choice=material_choice,
            finish_choice=finish_choice,
            condition_choice=condition_choice,
        )

        st.caption(f"Puntaje equivalente para el modelo: {inferred_score}/10")
        if self.show_rule_details:
            with st.expander("Ver reglas activadas", expanded=False):
                for rule in rules:
                    st.write(f"- {rule}")

        return inferred_score

    def render_overall_condition_widget(
        self,
        feature: str,
        default_value: object,
        key_prefix: str,
    ) -> int:
        label = FeatureCatalog.label(feature)
        help_text = FeatureCatalog.help_text(feature)
        default_score = int(round(InputPreparationService.to_float(default_value, 6.0)))
        default_idx = FeatureInputRenderer._nearest_choice_index(OVERALL_CONDITION_CHOICES, default_score)

        selected = st.selectbox(
            f"{label} (seleccion guiada)",
            options=OVERALL_CONDITION_CHOICES,
            index=default_idx,
            format_func=lambda option: f"{option[0]} ({option[1]}/10)",
            key=f"{key_prefix}_overallcond",
            help=help_text,
        )

        score = int(selected[1])
        st.caption(f"Valor equivalente para el modelo: {score}/10")
        return score

    def render_category_quality_expert_widget(
        self,
        feature: str,
        default_value: object,
        key_prefix: str,
    ) -> str:
        ruleset = QualityExpertEngine.ruleset(feature)
        if not ruleset:
            return str(default_value) if default_value is not None else "TA"

        label = FeatureCatalog.label(feature)
        help_text = FeatureCatalog.help_text(feature)
        intro = str(ruleset.get("intro", ""))

        material_options, finish_options, condition_options = QualityExpertEngine.category_options(feature)
        default_code = str(default_value) if default_value is not None else "TA"
        material_default, finish_default, condition_default = QualityExpertEngine.default_category_profile(feature, default_code)

        st.caption(f"{label}: {intro}")

        material_choice = st.selectbox(
            str(ruleset["material_question"]),
            options=material_options,
            index=material_options.index(material_default),
            key=f"{key_prefix}_{feature}_material",
        )
        finish_choice = st.selectbox(
            str(ruleset["finish_question"]),
            options=finish_options,
            index=finish_options.index(finish_default),
            key=f"{key_prefix}_{feature}_finish",
        )
        condition_choice = st.selectbox(
            str(ruleset["condition_question"]),
            options=condition_options,
            index=condition_options.index(condition_default),
            key=f"{key_prefix}_{feature}_condition",
            help=help_text,
        )

        inferred_code, inferred_level, rules = QualityExpertEngine.infer_quality_code(
            feature=feature,
            material_choice=material_choice,
            finish_choice=finish_choice,
            condition_choice=condition_choice,
        )

        if inferred_code == "NoBasement":
            st.caption("Codigo equivalente para el modelo: Sin sotano (NoBasement)")
        else:
            st.caption(
                "Codigo equivalente para el modelo: "
                f"{FeatureCatalog.format_option(feature, inferred_code)}"
                f" | Nivel {inferred_level}/5"
            )

        if self.show_rule_details:
            with st.expander("Ver reglas activadas", expanded=False):
                for rule in rules:
                    st.write(f"- {rule}")

        return inferred_code

    def render_feature_widget(
        self,
        feature: str,
        default_value: object,
        key_prefix: str,
    ) -> object:
        display_label = FeatureCatalog.label(feature)
        help_text = FeatureCatalog.help_text(feature)

        if feature == "OverallQual":
            return self.render_overall_quality_expert_widget(feature=feature, default_value=default_value, key_prefix=key_prefix)

        if feature == "OverallCond":
            return self.render_overall_condition_widget(feature=feature, default_value=default_value, key_prefix=key_prefix)

        if feature in QUALITY_CATEGORY_RULESETS:
            return self.render_category_quality_expert_widget(feature=feature, default_value=default_value, key_prefix=key_prefix)

        if self.ref_df is not None and feature in self.ref_df.columns:
            series = self.ref_df[feature]
            if pd.api.types.is_numeric_dtype(series):
                min_value, max_value = InputPreparationService.numeric_bounds(series)
                if default_value is not None:
                    value = InputPreparationService.to_float(default_value, InputPreparationService.numeric_default(series))
                else:
                    value = InputPreparationService.numeric_default(series)
                value = max(min_value, min(value, max_value))

                if feature in INTEGER_LIKE_FEATURES:
                    return st.number_input(
                        display_label,
                        min_value=int(math.floor(min_value)),
                        max_value=int(math.ceil(max_value)),
                        value=int(round(value)),
                        step=1,
                        help=help_text,
                        key=f"{key_prefix}_num_{feature}",
                    )

                return st.number_input(
                    display_label,
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(value),
                    step=1.0,
                    help=help_text,
                    key=f"{key_prefix}_num_{feature}",
                )

            options = sorted(series.dropna().astype(str).unique().tolist())
            if not options:
                options = ["Missing"]
            default_text = str(default_value) if default_value is not None else InputPreparationService.categorical_default(series)
            if default_text not in options:
                options = [default_text] + options

            return st.selectbox(
                display_label,
                options=options,
                index=options.index(default_text),
                format_func=lambda option: FeatureCatalog.format_option(feature, str(option)),
                help=help_text,
                key=f"{key_prefix}_cat_{feature}",
            )

        if feature in NUMERIC_FEATURE_HINTS:
            fallback = InputPreparationService.to_float(default_value, 0.0) if default_value is not None else 0.0
            return st.number_input(
                display_label,
                value=fallback,
                step=1.0,
                help=help_text,
                key=f"{key_prefix}_num_{feature}",
            )

        return st.text_input(
            display_label,
            value=str(default_value) if default_value is not None else "Missing",
            help=help_text,
            key=f"{key_prefix}_txt_{feature}",
        )

    def render_feature_grid(
        self,
        features: list[str],
        defaults: dict[str, object],
        key_prefix: str,
    ) -> dict[str, object]:
        values: dict[str, object] = {}
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                values[feature] = self.render_feature_widget(
                    feature=feature,
                    default_value=defaults.get(feature),
                    key_prefix=key_prefix,
                )
        return values

    def render_grouped_features(
        self,
        groups: list[tuple[str, list[str]]],
        defaults: dict[str, object],
        key_prefix: str,
    ) -> dict[str, object]:
        values: dict[str, object] = {}
        tab_titles = [title for title, _ in groups]
        tabs = st.tabs(tab_titles)

        for tab, (title, feature_list) in zip(tabs, groups):
            with tab:
                current = self.render_feature_grid(
                    features=feature_list,
                    defaults=defaults,
                    key_prefix=f"{key_prefix}_{InputPreparationService.slugify(title)}",
                )
                values.update(current)

        return values
