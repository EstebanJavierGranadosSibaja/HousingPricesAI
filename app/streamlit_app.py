"""Streamlit app for interactive housing price prediction using all manifest features."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import build_prediction_frame_from_dict, load_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_comparison.csv"

PRIMARY_INPUTS = ["GrLivArea", "Neighborhood", "BedroomAbvGr"]
INTEGER_LIKE_FEATURES = {
    "YearBuilt",
    "YearRemodAdd",
    "OverallQual",
    "OverallCond",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageYrBlt",
    "Fireplaces",
}
NUMERIC_FEATURE_HINTS = {
    "1stFlrSF",
    "2ndFlrSF",
    "BsmtFinSF1",
    "BsmtUnfSF",
    "Fireplaces",
    "GrLivArea",
    "LotArea",
    "LotFrontage",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageArea",
    "GarageYrBlt",
    "OpenPorchSF",
    "OverallCond",
    "OverallQual",
    "MasVnrArea",
    "TotalBsmtSF",
    "WoodDeckSF",
}
FEATURE_GROUP_BLUEPRINT = [
    (
        "Ubicacion y lote",
        ["MSZoning", "Neighborhood", "LotArea", "LotFrontage"],
    ),
    (
        "Calidad y condicion",
        ["OverallQual", "OverallCond", "ExterQual", "SaleCondition", "Foundation"],
    ),
    (
        "Espacios interiores",
        [
            "GrLivArea",
            "1stFlrSF",
            "2ndFlrSF",
            "TotRmsAbvGrd",
            "BedroomAbvGr",
            "KitchenQual",
            "FullBath",
            "HalfBath",
        ],
    ),
    (
        "Sotano",
        ["TotalBsmtSF", "BsmtFinSF1", "BsmtUnfSF", "BsmtQual"],
    ),
    (
        "Garage y exterior",
        [
            "GarageCars",
            "GarageArea",
            "GarageYrBlt",
            "GarageFinish",
            "OpenPorchSF",
            "WoodDeckSF",
            "MasVnrArea",
            "MasVnrType",
            "Fireplaces",
        ],
    ),
    (
        "Tiempo",
        ["YearBuilt", "YearRemodAdd"],
    ),
]


@st.cache_resource
def load_models():
    linear_path = MODELS_DIR / "linear_regression.joblib"
    rf_path = MODELS_DIR / "random_forest.joblib"

    if not linear_path.exists() or not rf_path.exists():
        return None, None

    return joblib.load(linear_path), joblib.load(rf_path)


@st.cache_data
def load_reference_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    if not METRICS_PATH.exists():
        return None
    df = pd.read_csv(METRICS_PATH)
    if "RMSE" in df.columns:
        df = df.sort_values(by="RMSE", ascending=True)
    return df


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0a0a0a;
            --sidebar: #121212;
            --panel: #161616;
            --line: #2a2a2a;
            --ink: #f4f4f4;
            --muted: #b3b3b3;
            --accent: #7d7d7d;
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar);
            border-right: 1px solid var(--line);
        }
        .main .block-container {
            padding-top: 0.8rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }
        .hero-box {
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.95rem 1rem;
            margin-bottom: 1rem;
            background: var(--panel);
        }
        .hero-box h2 {
            color: var(--ink);
            letter-spacing: 0.2px;
            margin: 0;
        }
        .muted {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }
        .note-card {
            border-left: 3px solid var(--accent);
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #171717;
            padding: 0.55rem 0.7rem;
            margin-bottom: 0.55rem;
            color: var(--ink);
        }
        .model-pill {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #171717;
            padding: 0.38rem 0.5rem;
            margin-bottom: 0.3rem;
        }
        .model-pill .title {
            color: #e8e8e8;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.08rem;
        }
        .model-pill .meta {
            color: #b0b0b0;
            font-size: 0.73rem;
            line-height: 1.2;
        }
        [data-testid="stForm"] {
            background: #141414;
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.85rem 0.95rem 0.7rem 0.95rem;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            background: #131313;
        }
        [data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.45rem 0.55rem;
            background: #141414;
        }
        [data-testid="stMetricLabel"] {
            color: #bdbdbd;
        }
        [data-testid="stMetricValue"] {
            color: #f6f6f6;
        }
        [data-testid="stProgress"] > div > div > div > div {
            background: #8a8a8a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _numeric_default(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().any():
        return float(values.median())
    return 0.0


def _categorical_default(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if not values.empty:
        return str(values.mode().iloc[0])
    return "Missing"


def _numeric_bounds(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0, 100.0
    low = float(values.quantile(0.01))
    high = float(values.quantile(0.99))
    if low == high:
        high = low + 1.0
    return low, high


def _series_quantile(ref_df: pd.DataFrame | None, column: str, q: float, fallback: float) -> float:
    if ref_df is None or column not in ref_df.columns:
        return fallback
    values = pd.to_numeric(ref_df[column], errors="coerce").dropna()
    if values.empty:
        return fallback
    return float(values.quantile(q))


def _series_mode(ref_df: pd.DataFrame | None, column: str, fallback: str, rank: int = 0) -> str:
    if ref_df is None or column not in ref_df.columns:
        return fallback
    counts = ref_df[column].dropna().astype(str).value_counts()
    if counts.empty:
        return fallback
    rank = min(rank, len(counts) - 1)
    return str(counts.index[rank])


def build_defaults(features: list[str], ref_df: pd.DataFrame | None) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for feature in features:
        if ref_df is not None and feature in ref_df.columns:
            series = ref_df[feature]
            if pd.api.types.is_numeric_dtype(series):
                defaults[feature] = _numeric_default(series)
            else:
                defaults[feature] = _categorical_default(series)
        else:
            defaults[feature] = 0.0 if feature in NUMERIC_FEATURE_HINTS else "Missing"
    return defaults


def apply_preset(
    preset: str,
    defaults: dict[str, object],
    ref_df: pd.DataFrame | None,
) -> dict[str, object]:
    values = dict(defaults)

    if preset == "Base (mediana/moda)" or ref_df is None:
        return values

    if preset == "Casa compacta":
        values["GrLivArea"] = _series_quantile(ref_df, "GrLivArea", 0.25, float(values.get("GrLivArea", 1200.0)))
        values["LotArea"] = _series_quantile(ref_df, "LotArea", 0.25, float(values.get("LotArea", 8000.0)))
        values["OverallQual"] = round(_series_quantile(ref_df, "OverallQual", 0.35, 5.0))
        values["BedroomAbvGr"] = round(_series_quantile(ref_df, "BedroomAbvGr", 0.40, 2.0))
        values["GarageCars"] = round(_series_quantile(ref_df, "GarageCars", 0.35, 1.0))
        values["Neighborhood"] = _series_mode(ref_df, "Neighborhood", str(values.get("Neighborhood", "NAmes")), rank=2)
        return values

    if preset == "Casa familiar":
        values["GrLivArea"] = _series_quantile(ref_df, "GrLivArea", 0.55, float(values.get("GrLivArea", 1500.0)))
        values["LotArea"] = _series_quantile(ref_df, "LotArea", 0.55, float(values.get("LotArea", 9000.0)))
        values["OverallQual"] = round(_series_quantile(ref_df, "OverallQual", 0.60, 6.0))
        values["BedroomAbvGr"] = round(_series_quantile(ref_df, "BedroomAbvGr", 0.65, 3.0))
        values["GarageCars"] = round(_series_quantile(ref_df, "GarageCars", 0.60, 2.0))
        values["Neighborhood"] = _series_mode(ref_df, "Neighborhood", str(values.get("Neighborhood", "NAmes")), rank=0)
        return values

    if preset == "Casa premium":
        values["GrLivArea"] = _series_quantile(ref_df, "GrLivArea", 0.90, float(values.get("GrLivArea", 2300.0)))
        values["LotArea"] = _series_quantile(ref_df, "LotArea", 0.88, float(values.get("LotArea", 12000.0)))
        values["OverallQual"] = max(8, round(_series_quantile(ref_df, "OverallQual", 0.90, 8.0)))
        values["BedroomAbvGr"] = max(4, round(_series_quantile(ref_df, "BedroomAbvGr", 0.85, 4.0)))
        values["GarageCars"] = max(2, round(_series_quantile(ref_df, "GarageCars", 0.85, 2.0)))
        values["Neighborhood"] = _series_mode(ref_df, "Neighborhood", str(values.get("Neighborhood", "NridgHt")), rank=1)
        return values

    return values


def resolve_feature_groups(features: list[str]) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    used: set[str] = set()

    for title, candidates in FEATURE_GROUP_BLUEPRINT:
        current = [feature for feature in candidates if feature in features and feature not in used]
        if current:
            groups.append((title, current))
            used.update(current)

    remaining = [feature for feature in features if feature not in used]
    if remaining:
        groups.append(("Otros", remaining))

    return groups


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text)


def render_metrics_sidebar(metrics_df: pd.DataFrame | None) -> None:
    if metrics_df is None or metrics_df.empty:
        return

    st.subheader("Rendimiento historico")
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

        meta_text = " | ".join(meta_chunks)
        st.markdown(
            f"""
            <div class="model-pill">
                <div class="title">{model}</div>
                <div class="meta">{meta_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _to_float(value: object, fallback: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return fallback
    return float(parsed)


def _clip_numeric_feature(feature: str, value: float, ref_df: pd.DataFrame | None) -> float:
    if ref_df is not None and feature in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[feature]):
        low, high = _numeric_bounds(ref_df[feature])
        value = max(low, min(value, high))
    return value


def build_ab_scenario(
    base_values: dict[str, object],
    ref_df: pd.DataFrame | None,
    area_delta_pct: float,
    quality_delta: int,
    garage_delta: int,
    remodel_delta_years: int,
) -> tuple[dict[str, object], list[str]]:
    updated = dict(base_values)
    changes: list[str] = []

    if "GrLivArea" in updated and area_delta_pct != 0:
        current = _to_float(updated["GrLivArea"], 0.0)
        target = current * (1.0 + area_delta_pct / 100.0)
        target = _clip_numeric_feature("GrLivArea", target, ref_df)
        updated["GrLivArea"] = round(target, 1)
        changes.append(f"GrLivArea: {current:,.0f} -> {target:,.0f} m2")

    if "OverallQual" in updated and quality_delta != 0:
        current = _to_float(updated["OverallQual"], 5.0)
        target = _clip_numeric_feature("OverallQual", current + quality_delta, ref_df)
        target = int(round(target))
        updated["OverallQual"] = target
        changes.append(f"OverallQual: {int(round(current))} -> {target}")

    if "GarageCars" in updated and garage_delta != 0:
        current = _to_float(updated["GarageCars"], 1.0)
        target = _clip_numeric_feature("GarageCars", current + garage_delta, ref_df)
        target = int(round(target))
        updated["GarageCars"] = target
        changes.append(f"GarageCars: {int(round(current))} -> {target}")

        if "GarageArea" in updated:
            base_area = _to_float(updated["GarageArea"], 0.0)
            denom = max(current, 1.0)
            scaled_area = base_area * (max(target, 0.0) / denom)
            scaled_area = _clip_numeric_feature("GarageArea", scaled_area, ref_df)
            updated["GarageArea"] = round(scaled_area, 1)

    if "YearRemodAdd" in updated and remodel_delta_years != 0:
        current = _to_float(updated["YearRemodAdd"], 2000.0)
        target = _clip_numeric_feature("YearRemodAdd", current + remodel_delta_years, ref_df)
        target = int(round(target))
        updated["YearRemodAdd"] = target
        changes.append(f"YearRemodAdd: {int(round(current))} -> {target}")

    return updated, changes


def render_feature_widget(
    feature: str,
    ref_df: pd.DataFrame | None,
    default_value: object,
    key_prefix: str,
) -> object:
    if ref_df is not None and feature in ref_df.columns:
        series = ref_df[feature]
        if pd.api.types.is_numeric_dtype(series):
            min_value, max_value = _numeric_bounds(series)
            value = float(default_value) if default_value is not None else _numeric_default(series)
            value = max(min_value, min(value, max_value))

            if feature in INTEGER_LIKE_FEATURES:
                return st.number_input(
                    feature,
                    min_value=int(math.floor(min_value)),
                    max_value=int(math.ceil(max_value)),
                    value=int(round(value)),
                    step=1,
                    key=f"{key_prefix}_num_{feature}",
                )

            return st.number_input(
                feature,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(value),
                step=1.0,
                key=f"{key_prefix}_num_{feature}",
            )

        options = sorted(series.dropna().astype(str).unique().tolist())
        if not options:
            options = ["Missing"]
        default_text = str(default_value) if default_value is not None else _categorical_default(series)
        if default_text not in options:
            options = [default_text] + options
        return st.selectbox(
            feature,
            options=options,
            index=options.index(default_text),
            key=f"{key_prefix}_cat_{feature}",
        )

    if feature in NUMERIC_FEATURE_HINTS:
        fallback = float(default_value) if default_value is not None else 0.0
        return st.number_input(
            feature,
            value=fallback,
            step=1.0,
            key=f"{key_prefix}_num_{feature}",
        )

    return st.text_input(
        feature,
        value=str(default_value) if default_value is not None else "Missing",
        key=f"{key_prefix}_txt_{feature}",
    )


def render_feature_grid(
    features: list[str],
    ref_df: pd.DataFrame | None,
    defaults: dict[str, object],
    key_prefix: str,
) -> dict[str, object]:
    values: dict[str, object] = {}
    cols = st.columns(2)
    for idx, feature in enumerate(features):
        with cols[idx % 2]:
            values[feature] = render_feature_widget(
                feature=feature,
                ref_df=ref_df,
                default_value=defaults.get(feature),
                key_prefix=key_prefix,
            )
    return values


def render_grouped_features(
    groups: list[tuple[str, list[str]]],
    ref_df: pd.DataFrame | None,
    defaults: dict[str, object],
    key_prefix: str,
) -> dict[str, object]:
    values: dict[str, object] = {}
    tab_titles = [title for title, _ in groups]
    tabs = st.tabs(tab_titles)

    for tab, (title, feature_list) in zip(tabs, groups):
        with tab:
            current = render_feature_grid(
                features=feature_list,
                ref_df=ref_df,
                defaults=defaults,
                key_prefix=f"{key_prefix}_{slugify(title)}",
            )
            values.update(current)

    return values


def weighted_consensus(pred_linear: float, pred_rf: float, metrics_df: pd.DataFrame | None) -> float:
    if metrics_df is None or metrics_df.empty or "Model" not in metrics_df.columns or "RMSE" not in metrics_df.columns:
        return (pred_linear + pred_rf) / 2.0

    rmse_map = dict(zip(metrics_df["Model"], metrics_df["RMSE"]))
    rmse_linear = float(rmse_map.get("linear_regression", 1.0))
    rmse_rf = float(rmse_map.get("random_forest", 1.0))
    w_linear = 1.0 / max(rmse_linear, 1e-9)
    w_rf = 1.0 / max(rmse_rf, 1e-9)
    return (pred_linear * w_linear + pred_rf * w_rf) / (w_linear + w_rf)


def disagreement_level(
    pred_linear: float,
    pred_rf: float,
    metrics_df: pd.DataFrame | None,
) -> tuple[str, str, float, float]:
    diff_abs = abs(pred_linear - pred_rf)
    mean_pred = max((pred_linear + pred_rf) / 2.0, 1.0)
    ratio = diff_abs / mean_pred

    rmse_ref = 0.0
    if metrics_df is not None and not metrics_df.empty and "RMSE" in metrics_df.columns:
        rmse_ref = float(metrics_df["RMSE"].mean())

    if ratio < 0.10 and (rmse_ref == 0.0 or diff_abs <= 1.5 * rmse_ref):
        return "baja", "La diferencia esta dentro del rango esperado entre modelos.", diff_abs, ratio
    if ratio < 0.25 and (rmse_ref == 0.0 or diff_abs <= 3.0 * rmse_ref):
        return "media", "Hay diferencia moderada, conviene revisar valores de entrada.", diff_abs, ratio
    return "alta", "La diferencia es alta, puede haber combinaciones atipicas o alta no linealidad.", diff_abs, ratio


def percentile_rank(series: pd.Series, value: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values <= value).mean() * 100.0)


def detect_extreme_inputs(sample: pd.DataFrame, ref_df: pd.DataFrame | None) -> list[str]:
    if ref_df is None or sample.empty:
        return []

    watch_features = ["GrLivArea", "LotArea", "OverallQual", "GarageArea", "YearBuilt"]
    row = sample.iloc[0]
    notes: list[str] = []

    for feature in watch_features:
        if feature not in row.index or feature not in ref_df.columns:
            continue

        value = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue

        p = percentile_rank(ref_df[feature], float(value))
        if p is None:
            continue

        if p <= 5.0 or p >= 95.0:
            notes.append(f"{feature} esta en percentil {p:.1f}%")

    return notes


def main() -> None:
    st.set_page_config(page_title="Prediccion de Viviendas", page_icon="house", layout="wide")
    inject_styles()

    st.markdown(
        """
        <div class="hero-box">
            <h2 style="margin:0;">Prediccion de Precio de Viviendas</h2>
            <p class="muted" style="margin-top:0.35rem; margin-bottom:0;">
                Flujo minimalista para estimar precio y comparar escenarios de mejora.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    linear_model, rf_model = load_models()
    if linear_model is None or rf_model is None:
        st.error("No se encontraron modelos entrenados. Ejecuta: python -m src.train")
        st.stop()

    features = load_features()
    groups = resolve_feature_groups(features)
    ref_df = load_reference_data()
    metrics_df = load_metrics()

    base_defaults = build_defaults(features, ref_df)

    with st.sidebar:
        st.header("Panel de control")
        mode = st.radio(
            "Modo de entrada",
            [
                "Rapido (3 variables + defaults)",
                "Completo (editar todas las variables)",
            ],
            index=0,
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

        st.caption(f"Features activas: {len(features)}")

        render_metrics_sidebar(metrics_df)

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
                        "Cambio area util (%)",
                        min_value=-25,
                        max_value=40,
                        value=10,
                        step=1,
                    )
                )
                quality_delta = int(
                    st.slider(
                        "Cambio calidad (OverallQual)",
                        min_value=-2,
                        max_value=2,
                        value=1,
                        step=1,
                    )
                )
                garage_delta = int(
                    st.slider(
                        "Cambio garage (GarageCars)",
                        min_value=-1,
                        max_value=2,
                        value=1,
                        step=1,
                    )
                )
                remodel_delta_years = int(
                    st.slider(
                        "Remodelacion adicional (anios)",
                        min_value=0,
                        max_value=25,
                        value=5,
                        step=1,
                    )
                )

    defaults = apply_preset(preset, base_defaults, ref_df)

    st.subheader("Entradas")
    if ref_df is None:
        st.warning("No se encontro train.csv, se usaran defaults genericos donde aplique.")
    else:
        st.caption("Defaults y presets calculados desde train.csv.")

    with st.form("predict_form", clear_on_submit=False):
        if mode.startswith("Rapido"):
            st.write("Configura 3 variables clave.")
            quick_defaults = {k: defaults[k] for k in PRIMARY_INPUTS}
            quick_values = render_feature_grid(
                features=PRIMARY_INPUTS,
                ref_df=ref_df,
                defaults=quick_defaults,
                key_prefix="quick",
            )

            full_values = dict(defaults)
            full_values.update(quick_values)

            with st.expander("Ajustes avanzados", expanded=False):
                optional_features = [f for f in features if f not in PRIMARY_INPUTS]
                optional_groups: list[tuple[str, list[str]]] = []
                for title, group_features in groups:
                    extra = [feature for feature in group_features if feature in optional_features]
                    if extra:
                        optional_groups.append((title, extra))

                optional_defaults = {feature: full_values[feature] for feature in optional_features}
                optional_values = render_grouped_features(
                    groups=optional_groups,
                    ref_df=ref_df,
                    defaults=optional_defaults,
                    key_prefix="optional",
                )
                full_values.update(optional_values)
        else:
            st.write("Modo completo: puedes editar todo el vector de entrada.")
            full_values = render_grouped_features(
                groups=groups,
                ref_df=ref_df,
                defaults=defaults,
                key_prefix="full",
            )

        submitted = st.form_submit_button("Predecir")

    if not submitted:
        return

    sample_values = dict(defaults)
    sample_values.update(full_values)

    sample = build_prediction_frame_from_dict(sample_values)
    pred_linear = float(linear_model.predict(sample)[0])
    pred_rf = float(rf_model.predict(sample)[0])
    pred_consensus = weighted_consensus(pred_linear, pred_rf, metrics_df)

    compare_values: dict[str, object] | None = None
    compare_changes: list[str] = []
    pred_linear_b = pred_rf_b = pred_consensus_b = None
    if compare_mode:
        compare_values, compare_changes = build_ab_scenario(
            base_values=sample_values,
            ref_df=ref_df,
            area_delta_pct=area_delta_pct,
            quality_delta=quality_delta,
            garage_delta=garage_delta,
            remodel_delta_years=remodel_delta_years,
        )
        sample_b = build_prediction_frame_from_dict(compare_values)
        pred_linear_b = float(linear_model.predict(sample_b)[0])
        pred_rf_b = float(rf_model.predict(sample_b)[0])
        pred_consensus_b = weighted_consensus(pred_linear_b, pred_rf_b, metrics_df)

    level, level_text, diff_abs, diff_ratio = disagreement_level(pred_linear, pred_rf, metrics_df)

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

    c1, c2, c3 = st.columns(3)
    c1.metric("Regresion Lineal", f"${pred_linear:,.2f}")
    c2.metric("Random Forest", f"${pred_rf:,.2f}")
    c3.metric("Estimacion consenso", f"${pred_consensus:,.2f}")

    st.caption(f"Cobertura de features en inferencia: {int(sample.notna().sum(axis=1).iloc[0])}/{len(features)}")

    with st.expander("Comparacion de modelos", expanded=True):
        if level == "baja":
            st.success(f"Diferencia entre modelos: ${diff_abs:,.2f} ({diff_ratio:.1%}). {level_text}")
        elif level == "media":
            st.warning(f"Diferencia entre modelos: ${diff_abs:,.2f} ({diff_ratio:.1%}). {level_text}")
        else:
            st.error(f"Diferencia entre modelos: ${diff_abs:,.2f} ({diff_ratio:.1%}). {level_text}")

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

        if compare_changes:
            st.caption("Cambios aplicados en escenario B:")
            for change in compare_changes:
                st.write(f"- {change}")

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

    with st.expander("Diagnostico", expanded=False):
        st.write(
            "- Regresion Lineal captura tendencia global, Random Forest captura patrones no lineales por segmentos."
        )
        st.write(
            "- Diferencias altas aparecen cuando la combinacion de features es rara o esta en los bordes del dataset."
        )
        st.write(
            "- Si la diferencia sube mucho, revisa primero Neighborhood, OverallQual, GrLivArea y GarageArea."
        )

        extreme_notes = detect_extreme_inputs(sample, ref_df)
        if extreme_notes:
            st.caption("Variables en zonas extremas del dataset:")
            for note in extreme_notes:
                st.write(f"- {note}")

        if ref_df is not None and "SalePrice" in ref_df.columns:
            p_linear = percentile_rank(ref_df["SalePrice"], pred_linear)
            p_rf = percentile_rank(ref_df["SalePrice"], pred_rf)
            p_cons = percentile_rank(ref_df["SalePrice"], pred_consensus)

            st.caption("Contexto percentil en distribucion historica:")
            if p_linear is not None:
                st.write(f"- Regresion Lineal: {p_linear:.1f}%")
            if p_rf is not None:
                st.write(f"- Random Forest: {p_rf:.1f}%")
            if p_cons is not None:
                st.write(f"- Consenso: {p_cons:.1f}%")

    history_df = pd.DataFrame(st.session_state.get("prediction_history", []))
    if len(history_df) >= 2:
        with st.expander("Historial de corridas", expanded=False):
            history_df = history_df.reset_index(drop=True)
            history_df.index = history_df.index + 1
            history_df.index.name = "Corrida"
            st.line_chart(history_df[["Regresion Lineal", "Random Forest", "Consenso"]])


if __name__ == "__main__":
    main()
