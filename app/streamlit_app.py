"""Streamlit app — wizard-style housing price predictor (6 steps)."""

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
from app.explainability import ExplanationService
from app.services import DataRepository, InputPreparationService
from app.ui import inject_styles, render_bars, render_header, render_hero, render_section, render_table
from app.validation import InputValidator
from src.preprocessing import build_prediction_frame_from_dict, load_features


@st.cache_resource
def _get_explanation_service() -> ExplanationService:
    return ExplanationService()


MODEL_LABELS = {"linear_regression": "Regresión Lineal", "random_forest": "Random Forest"}

SENTINEL = "__empty__"
SCALE_0_10 = {"OverallQual", "OverallCond"}
YEAR_FEATURES = {"YearBuilt", "YearRemodAdd"}
AREA_FEATURES = {"GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LotArea", "GarageArea"}

# Features where value=0 is the "not-yet-filled" sentinel for required fields.
ZERO_IS_EMPTY: set[str] = SCALE_0_10 | AREA_FEATURES | {"BedroomAbvGr"}

WIZARD_STEPS: list[dict] = [
    {
        "title": "Información básica",
        "subtitle": "Ubicación y tipo de zona de la vivienda",
        "icon": "📍",
        "short": "Básica",
        "fields": ["Neighborhood", "MSZoning", "LotArea"],
        "required": {"Neighborhood", "MSZoning"},
    },
    {
        "title": "Tamaño y distribución",
        "subtitle": "Superficie interior y distribución de espacios",
        "icon": "📐",
        "short": "Tamaño",
        "fields": [
            "GrLivArea", "BedroomAbvGr",
            "FullBath", "HalfBath", "TotRmsAbvGrd",
            "1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "Fireplaces",
        ],
        "required": {"GrLivArea", "BedroomAbvGr"},
    },
    {
        "title": "Calidad y acabados",
        "subtitle": "Nivel de construcción, materiales y equipamiento interior",
        "icon": "⭐",
        "short": "Calidad",
        "fields": ["OverallQual", "OverallCond", "KitchenQual", "ExterQual", "BsmtQual"],
        "required": {"OverallQual"},
    },
    {
        "title": "Garaje y antigüedad",
        "subtitle": "Infraestructura adicional y años de construcción",
        "icon": "🏗️",
        "short": "Garaje",
        "fields": ["YearBuilt", "YearRemodAdd", "GarageCars", "GarageArea", "Foundation"],
        "required": {"YearBuilt"},
    },
]

STEP_SUMMARY = 4
STEP_RESULT = 5
TOTAL_STEPS = 6
_STEP_SHORTS = [s["short"] for s in WIZARD_STEPS] + ["Resumen", "Resultado"]


# ── State helpers ─────────────────────────────────────────────────────────────

def _init_wizard() -> None:
    for key, default in [("wizard_step", 0), ("wizard_errors", []), ("_wizard_data", {})]:
        if key not in st.session_state:
            st.session_state[key] = default


def _wkey(feature: str) -> str:
    return f"wiz_{feature}"


def _is_empty(feature: str, val: object) -> bool:
    if val is None or val == SENTINEL:
        return True
    if feature in ZERO_IS_EMPTY:
        try:
            return int(val) == 0
        except (ValueError, TypeError):
            return True
    return False


def _save_step(step_idx: int) -> None:
    if step_idx >= len(WIZARD_STEPS):
        return
    store: dict = st.session_state["_wizard_data"]
    for feature in WIZARD_STEPS[step_idx]["fields"]:
        key = _wkey(feature)
        if key in st.session_state:
            store[feature] = st.session_state[key]


def _restore_step(step_idx: int) -> None:
    if step_idx >= len(WIZARD_STEPS):
        return
    store: dict = st.session_state["_wizard_data"]
    for feature in WIZARD_STEPS[step_idx]["fields"]:
        if feature in store:
            st.session_state[_wkey(feature)] = store[feature]


def _validate_step(step_idx: int) -> list[str]:
    if step_idx >= len(WIZARD_STEPS):
        return []
    return [
        FeatureCatalog.label(f)
        for f in WIZARD_STEPS[step_idx]["required"]
        if _is_empty(f, st.session_state.get(_wkey(f)))
    ]


def _all_required_filled() -> bool:
    store: dict = st.session_state.get("_wizard_data", {})
    return all(
        not _is_empty(f, store.get(f))
        for step_def in WIZARD_STEPS
        for f in step_def["required"]
    )


def _collect_values() -> dict[str, object]:
    store: dict = st.session_state.get("_wizard_data", {})
    return {f: v for f, v in store.items() if not _is_empty(f, v)}


def _reset_wizard() -> None:
    for step_def in WIZARD_STEPS:
        for feature in step_def["fields"]:
            st.session_state.pop(_wkey(feature), None)
    st.session_state.update(wizard_step=0, wizard_errors=[], _wizard_data={})
    for k in ("result", "explanation", "explanation_factors"):
        st.session_state.pop(k, None)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _bounds(ref_df: pd.DataFrame | None, feature: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if ref_df is not None and feature in ref_df.columns:
        lo, hi = InputPreparationService.numeric_bounds(ref_df[feature])
        return float(lo), float(hi)
    return fallback


def _quality_options(feature: str, ref_df: pd.DataFrame | None) -> list[str]:
    label_map = CATEGORICAL_OPTION_LABELS_BY_FEATURE.get(feature, {})
    data_opts: list[str] = []
    if ref_df is not None and feature in ref_df.columns:
        data_opts = sorted(ref_df[feature].dropna().astype(str).unique().tolist())
    if label_map:
        ordered = [c for c in label_map if c != "Missing" and (not data_opts or c in data_opts)]
        ordered += [c for c in data_opts if c not in ordered and c not in label_map]
        return ordered or [c for c in label_map if c != "Missing"]
    return data_opts or ["Missing"]


# ── CSS ───────────────────────────────────────────────────────────────────────

def _inject_wizard_styles() -> None:
    st.markdown(
        """
        <style>
        /* Step indicator */
        .wiz-track {
            display: flex; align-items: flex-start; margin: 0.2rem 0 1.5rem;
        }
        .wiz-node { display: flex; flex-direction: column; align-items: center; flex-shrink: 0; }
        .wiz-circle {
            width: 32px; height: 32px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.78rem; font-weight: 800; transition: all .2s;
        }
        .wiz-circle.done {
            background: linear-gradient(135deg,var(--accent),var(--accent2));
            color: #1a1407;
            box-shadow: 0 0 14px color-mix(in srgb,var(--accent) 40%,transparent);
        }
        .wiz-circle.active {
            background: transparent; border: 2.5px solid var(--accent); color: var(--accent);
            box-shadow: 0 0 16px color-mix(in srgb,var(--accent) 20%,transparent);
        }
        .wiz-circle.future { background: var(--track); color: var(--muted); }
        .wiz-node-lbl { font-size: 0.6rem; color: var(--muted); margin-top: 0.3rem; text-align: center; max-width: 54px; line-height: 1.2; }
        .wiz-node-lbl.active { color: var(--accent); font-weight: 700; }
        .wiz-conn { flex: 1; height: 2px; margin-top: 15px; background: var(--line); transition: background .3s; }
        .wiz-conn.done { background: linear-gradient(90deg,var(--accent2),var(--accent)); }

        /* Step header */
        .wiz-kicker { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.14em; color: var(--accent); font-weight: 700; margin-bottom: 0.1rem; }
        .wiz-title { font-size: 1.3rem; font-weight: 900; color: var(--ink); letter-spacing: -0.015em; margin: 0.05rem 0 0.1rem; }
        .wiz-sub { color: var(--muted); font-size: 0.88rem; margin-bottom: 0.2rem; }
        .wiz-divider { border: none; border-top: 1px solid var(--line); margin: 0.75rem 0 1rem; }

        /* Field sections */
        .wiz-req-hdr {
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 0.65rem;
        }
        .wiz-req-label {
            font-size: 0.69rem; text-transform: uppercase; letter-spacing: 0.14em;
            color: var(--accent); font-weight: 800;
            border-left: 3px solid var(--accent); padding-left: 0.5rem;
        }
        .wiz-req-count { font-size: 0.72rem; color: var(--muted); font-weight: 600; }
        .wiz-req-count.all-done { color: #4ade80; }
        .wiz-opt-label {
            font-size: 0.69rem; text-transform: uppercase; letter-spacing: 0.12em;
            color: var(--muted); font-weight: 600; margin-bottom: 0.55rem; margin-top: 0.4rem;
        }

        /* Inline field validation hint */
        .wiz-field-err { color: #f87171; font-size: 0.74rem; margin-top: -0.15rem; line-height: 1.4; }

        /* Error summary box */
        .wiz-err-box {
            background: rgba(239,68,68,.09); border: 1px solid rgba(239,68,68,.3);
            border-radius: 12px; padding: 0.75rem 1rem;
            color: #fca5a5; font-size: 0.86rem; margin: 0.6rem 0; line-height: 1.8;
        }

        /* Summary cards */
        .sum-card {
            background: var(--panel); border: 1px solid var(--line);
            border-radius: 16px; padding: 1rem 1.25rem 0.75rem;
            margin-bottom: 0.9rem; box-shadow: 0 6px 20px rgba(0,0,0,.14);
        }
        .sum-group-title { font-size: 0.95rem; font-weight: 800; color: var(--ink); margin-bottom: 0.65rem; }
        .sum-row {
            display: flex; justify-content: space-between; align-items: center;
            padding: 0.33rem 0; border-bottom: 1px solid var(--line); gap: 0.6rem;
        }
        .sum-row:last-child { border-bottom: none; }
        .sum-lbl { color: var(--muted); font-size: 0.83rem; flex-shrink: 0; max-width: 54%; }
        .sum-val { color: var(--ink); font-weight: 700; font-size: 0.83rem; text-align: right; }
        .sum-empty { color: var(--muted); font-size: 0.83rem; font-style: italic; text-align: right; }
        .sum-miss { color: #f87171; font-size: 0.83rem; font-weight: 700; text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Step indicator + header ───────────────────────────────────────────────────

def _render_indicator(step: int) -> None:
    html = ""
    for i, name in enumerate(_STEP_SHORTS):
        cls = "done" if i < step else ("active" if i == step else "future")
        num = "✓" if i < step else str(i + 1)
        lbl_cls = "active" if i == step else ""
        html += (
            f"<div class='wiz-node'>"
            f"<div class='wiz-circle {cls}'>{num}</div>"
            f"<div class='wiz-node-lbl {lbl_cls}'>{name}</div>"
            f"</div>"
        )
        if i < len(_STEP_SHORTS) - 1:
            conn_cls = "done" if i < step else ""
            html += f"<div class='wiz-conn {conn_cls}'></div>"
    st.markdown(f"<div class='wiz-track'>{html}</div>", unsafe_allow_html=True)


def _render_step_header(step: int) -> None:
    if step < len(WIZARD_STEPS):
        s = WIZARD_STEPS[step]
        icon, title, sub = s["icon"], s["title"], s["subtitle"]
    elif step == STEP_SUMMARY:
        icon, title, sub = "📋", "Resumen y validación", "Verifica los datos antes de calcular el precio"
    else:
        icon, title, sub = "🏠", "Resultado", "Estimación de precio completada"

    st.markdown(
        f"<div class='wiz-kicker'>Paso {step + 1} de {TOTAL_STEPS}</div>"
        f"<div class='wiz-title'>{icon}&nbsp;{title}</div>"
        f"<div class='wiz-sub'>{sub}</div>"
        f"<hr class='wiz-divider'>",
        unsafe_allow_html=True,
    )


# ── Field renderer ────────────────────────────────────────────────────────────

def _inline_err(feature: str, key: str, required: bool, msg: str) -> None:
    """Show inline error hint only when: required + errors exist + field still empty."""
    if required and st.session_state.get("wizard_errors") and _is_empty(feature, st.session_state.get(key)):
        st.markdown(f"<div class='wiz-field-err'>⚠ {msg}</div>", unsafe_allow_html=True)


def _render_field(
    feature: str,
    required: bool,
    ref_df: pd.DataFrame | None,
    defaults: dict[str, object],
) -> None:
    label = FeatureCatalog.label(feature)
    help_text = FeatureCatalog.help_text(feature)
    key = _wkey(feature)
    lbl = f"{label} *" if required else label
    cur = st.session_state.get(key)

    # ── Quality / condition scale (0 = not selected) ──────────────
    if feature in SCALE_0_10:
        opts = list(range(0, 11))
        try:
            init = int(cur) if cur is not None and int(cur) in opts else 0
        except (ValueError, TypeError):
            init = 0
        _tags = {0: "— Seleccione —", 1: " — Muy básica", 5: " — Promedio",
                 8: " — Muy buena", 10: " — Excelente"}
        st.select_slider(
            lbl, options=opts, value=init,
            format_func=lambda v: "— Seleccione —" if v == 0 else f"{v}{_tags.get(v, '')}",
            help=help_text or "Arrastra para seleccionar de 1 (muy básica) a 10 (excelente)",
            key=key,
        )
        _inline_err(feature, key, required, "Mueve el selector para indicar la calidad")
        return

    # ── Categorical ───────────────────────────────────────────────
    is_cat = (
        ref_df is not None and feature in ref_df.columns
        and not pd.api.types.is_numeric_dtype(ref_df[feature])
    )
    if is_cat or feature in CATEGORICAL_OPTION_LABELS_BY_FEATURE:
        real_opts = _quality_options(feature, ref_df)
        lmap = CATEGORICAL_OPTION_LABELS_BY_FEATURE.get(feature, {})
        if required:
            all_opts = [SENTINEL] + real_opts
            idx = all_opts.index(cur) if cur in all_opts else 0
            st.selectbox(
                lbl, all_opts, index=idx,
                format_func=lambda c: "— Seleccione —" if c == SENTINEL else lmap.get(c, c),
                help=help_text, key=key,
            )
            _inline_err(feature, key, required, "Elige una opción de la lista")
        else:
            def_code = str(defaults.get(feature, real_opts[0]))
            if def_code not in real_opts:
                def_code = real_opts[0]
            idx = real_opts.index(cur) if cur in real_opts else real_opts.index(def_code)
            st.selectbox(
                lbl, real_opts, index=idx,
                format_func=lambda c: lmap.get(c, c),
                help=help_text, key=key,
            )
        return

    # ── Year ──────────────────────────────────────────────────────
    if feature in YEAR_FEATURES:
        lo, hi = _bounds(ref_df, feature, (1872.0, 2025.0))
        y_lo, y_hi = int(lo), int(hi)
        if required:
            try:
                init_v: int | None = int(cur) if cur is not None and cur != SENTINEL else None
            except (ValueError, TypeError):
                init_v = None
            st.number_input(
                lbl, min_value=y_lo, max_value=y_hi, value=init_v,
                step=1, help=help_text, key=key, placeholder=f"ej. {y_hi - 20}",
            )
            _inline_err(feature, key, required, f"Escribe el año ({y_lo} – {y_hi})")
        else:
            def_v = int(InputPreparationService.to_float(defaults.get(feature), (y_lo + y_hi) / 2))
            try:
                init_v = int(cur) if cur is not None and cur != SENTINEL else def_v
            except (ValueError, TypeError):
                init_v = def_v
            st.slider(lbl, y_lo, y_hi, init_v, step=1, help=help_text, key=key)
        return

    # ── Area (required starts at 0 so +/− buttons work) ──────────
    if feature in AREA_FEATURES:
        lo, hi = _bounds(ref_df, feature, (0.0, 5000.0))
        a_lo, a_hi = int(lo), int(hi)
        if required:
            try:
                init_v = int(cur) if cur is not None and cur != SENTINEL else 0
            except (ValueError, TypeError):
                init_v = 0
            st.number_input(
                lbl, min_value=0, max_value=a_hi, value=init_v,
                step=10, help=help_text, key=key,
            )
            cur_val = st.session_state.get(key, 0)
            if cur_val == 0:
                _inline_err(feature, key, required, f"Ingresa el área (rango típico: {a_lo:,}–{a_hi:,} ft²)")
            else:
                st.caption(f"Rango habitual: {a_lo:,} – {a_hi:,} ft²")
        else:
            def_v = int(InputPreparationService.to_float(defaults.get(feature), (a_lo + a_hi) / 2))
            try:
                init_v = int(cur) if cur is not None and cur != SENTINEL else def_v
            except (ValueError, TypeError):
                init_v = def_v
            step = max(10, int((a_hi - a_lo) / 100))
            st.slider(lbl, a_lo, a_hi, init_v, step=step, help=help_text, key=key)
        return

    # ── Integer-like ──────────────────────────────────────────────
    if feature in INTEGER_LIKE_FEATURES:
        lo, hi = _bounds(ref_df, feature, (0.0, 10.0))
        i_lo, i_hi = int(lo), int(hi)
        if required:
            try:
                init_v = int(cur) if cur is not None and cur != SENTINEL else 0
            except (ValueError, TypeError):
                init_v = 0
            st.number_input(
                lbl, min_value=0, max_value=i_hi, value=init_v,
                step=1, help=help_text, key=key,
            )
            _inline_err(feature, key, required, f"Ingresa la cantidad (mínimo 1)")
        else:
            def_v = int(round(InputPreparationService.to_float(defaults.get(feature), float(i_lo))))
            try:
                init_v = int(cur) if cur is not None and cur != SENTINEL else def_v
            except (ValueError, TypeError):
                init_v = def_v
            st.number_input(lbl, i_lo, i_hi, init_v, step=1, help=help_text, key=key)
        return

    # ── Generic numeric fallback ──────────────────────────────────
    lo, hi = _bounds(ref_df, feature, (0.0, 5000.0))
    def_v = InputPreparationService.to_float(defaults.get(feature), (lo + hi) / 2)
    try:
        init_v = float(cur) if cur is not None and cur != SENTINEL else def_v
    except (ValueError, TypeError):
        init_v = def_v
    st.number_input(lbl, float(lo), float(hi), init_v, step=1.0, help=help_text, key=key)


# ── Step layout ───────────────────────────────────────────────────────────────

def _render_step_fields(step_idx: int, ref_df: pd.DataFrame | None, defaults: dict[str, object]) -> None:
    req_set = WIZARD_STEPS[step_idx]["required"]
    all_fields = WIZARD_STEPS[step_idx]["fields"]
    req_fields = [f for f in all_fields if f in req_set]
    opt_fields = [f for f in all_fields if f not in req_set]

    if req_fields:
        filled = sum(1 for f in req_fields if not _is_empty(f, st.session_state.get(_wkey(f))))
        total = len(req_fields)
        count_cls = "all-done" if filled == total else ""
        count_txt = f"{filled}/{total} completados" if filled < total else "✓ Completados"
        st.markdown(
            f"<div class='wiz-req-hdr'>"
            f"<div class='wiz-req-label'>Campos obligatorios *</div>"
            f"<div class='wiz-req-count {count_cls}'>{count_txt}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(2, len(req_fields)))
        for i, feat in enumerate(req_fields):
            with cols[i % len(cols)]:
                _render_field(feat, True, ref_df, defaults)

    if opt_fields:
        st.write("")
        st.markdown(
            "<div class='wiz-opt-label'>Campos opcionales — se usan valores típicos si los omites</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, feat in enumerate(opt_fields):
            with cols[i % 2]:
                _render_field(feat, False, ref_df, defaults)


# ── Summary ───────────────────────────────────────────────────────────────────

def _fmt_val(feature: str, val: object) -> str:
    lmap = CATEGORICAL_OPTION_LABELS_BY_FEATURE.get(feature, {})
    if feature in SCALE_0_10:
        try:
            v = int(val)
            tags = {1: " — Muy básica", 5: " — Promedio", 8: " — Muy buena", 10: " — Excelente"}
            return f"{v}/10{tags.get(v, '')}"
        except (ValueError, TypeError):
            return str(val)
    if feature in AREA_FEATURES:
        try:
            return f"{int(val):,} ft²"
        except (ValueError, TypeError):
            return str(val)
    if feature in (YEAR_FEATURES | INTEGER_LIKE_FEATURES):
        try:
            return str(int(val))
        except (ValueError, TypeError):
            return str(val)
    return lmap.get(str(val), str(val)) if lmap else str(val)


def _render_summary() -> bool:
    render_section(
        "Resumen de la vivienda",
        "Verifica la información antes de calcular. Puedes editar cualquier paso.",
    )
    all_ready = _all_required_filled()
    if not all_ready:
        st.warning("Hay campos obligatorios sin completar — editálos antes de calcular.", icon="⚠️")

    store: dict = st.session_state.get("_wizard_data", {})

    for i, step_def in enumerate(WIZARD_STEPS):
        col_hdr, col_btn = st.columns([6, 1])
        with col_hdr:
            st.markdown(
                f"<div class='sum-group-title'>{step_def['icon']}&nbsp;{step_def['title']}</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("Editar", key=f"edit_{i}", use_container_width=True):
                _restore_step(i)
                st.session_state.update(wizard_step=i, wizard_errors=[])
                st.rerun()

        rows = ""
        for feat in step_def["fields"]:
            flabel = FeatureCatalog.label(feat)
            val = store.get(feat)
            is_req = feat in step_def["required"]
            is_mt = _is_empty(feat, val)
            if is_mt and is_req:
                v = "<span class='sum-miss'>⚠ Sin completar</span>"
            elif is_mt:
                v = "<span class='sum-empty'>valor típico</span>"
            else:
                v = f"<span class='sum-val'>{_fmt_val(feat, val)}</span>"
            rows += (
                f"<div class='sum-row'>"
                f"<span class='sum-lbl'>{flabel}</span>{v}"
                f"</div>"
            )
        st.markdown(f"<div class='sum-card'>{rows}</div>", unsafe_allow_html=True)

    return all_ready


# ── Compute ───────────────────────────────────────────────────────────────────

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
        "notes": notes, "percentile": percentile, "sample": sample,
    }


# ── Explanation tab ───────────────────────────────────────────────────────────

def _render_explanation_tab(result: dict[str, object], consensus: float, rf_model) -> None:
    render_section(
        "Explicación inteligente del precio",
        "Un LLM local (Ollama) analiza los factores SHAP y genera una explicación en lenguaje natural.",
    )
    service = _get_explanation_service()
    status = service.llm_status()
    if status["available"]:
        st.success(f"Ollama disponible · modelo: `{status['model']}`", icon="✅")
    else:
        st.warning(
            "Ollama no detectado en `localhost:11434`. "
            "Se usará explicación en plantilla. "
            "Instala Ollama y ejecuta `scripts/init_ollama.ps1` para activar la IA.",
            icon="⚠️",
        )
    sample = result.get("sample")
    if sample is None:
        st.info("Calcula un precio primero para obtener la explicación.")
        return
    if st.button("Generar explicación IA", key="btn_explain", type="primary"):
        with st.spinner("Analizando factores y generando explicación…"):
            explanation, factors = service.explain(sample, rf_model, consensus)
        st.session_state["explanation"] = explanation
        st.session_state["explanation_factors"] = factors
    explanation = st.session_state.get("explanation")
    factors = st.session_state.get("explanation_factors") or {}
    if explanation:
        st.markdown("---")
        st.markdown("#### Análisis de la estimación")
        st.markdown(explanation)
        pos = factors.get("positive", [])
        neg = factors.get("negative", [])
        if pos or neg:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Factores que elevan el precio**")
                for f in pos:
                    st.write(f"↑ {f['feature_label']}")
            with c2:
                st.markdown("**Factores que reducen el precio**")
                for f in neg:
                    st.write(f"↓ {f['feature_label']}")
    else:
        st.info(
            "Haz clic en **Generar explicación IA** para obtener un análisis "
            "en lenguaje natural de los factores que determinaron el precio."
        )


# ── Result page ───────────────────────────────────────────────────────────────

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
    t1, t2, t3, t4 = st.tabs(
        ["Comparación de modelos", "Variables más influyentes", "Rendimiento", "Explicación IA"]
    )
    with t1:
        render_section("Qué predice cada modelo", "La estimación final combina ambos, ponderando por su precisión histórica.")
        preds = [
            ("Regresión Lineal", float(result["pred_linear"]), False),
            ("Random Forest", float(result["pred_rf"]), False),
            ("Estimación combinada", float(result["consensus"]), True),
        ]
        mx = max(v for _, v, _ in preds) or 1.0
        render_bars([{"label": l, "value": f"${v:,.0f}", "frac": v / mx, "strong": s} for l, v, s in preds])
        if result.get("percentile") is not None:
            st.caption(f"Esta vivienda quedaría por encima del {float(result['percentile']):.0f}% del histórico.")
    with t2:
        render_section("Qué pesa más en el precio", "Importancia de variables aprendida por el Random Forest.")
        imp = PredictionAnalytics.feature_importances(rf_model, top_n=8)
        if imp is None or imp.empty:
            st.info("El modelo no expone importancia de variables.")
        else:
            mx = float(imp["Importancia"].max()) or 1.0
            render_bars([
                {"label": row.Variable, "value": f"{row.Importancia * 100:.0f}%",
                 "frac": row.Importancia / mx, "strong": i == 0}
                for i, row in enumerate(imp.itertuples())
            ])
            top = imp.iloc[0]
            st.caption(f"La variable más determinante es **{top['Variable']}** ({top['Importancia'] * 100:.0f}% del peso).")
    with t3:
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
            if "cv_rmse_mean" in metrics_df.columns:
                st.caption(
                    "Random Forest seleccionado como modelo operativo por su mejor desempeño "
                    "en test. La diferencia con Regresión Lineal no es estadísticamente "
                    "significativa (p ≈ 0.13, KFold=5). Ver documento técnico §5.1."
                )
    with t4:
        _render_explanation_tab(result, float(result["consensus"]), rf_model)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Estimador de Precio de Viviendas", page_icon="🏠", layout="wide"
    )
    _init_wizard()

    mode = st.session_state.get("theme_mode", "dark")
    col_hdr, col_toggle = st.columns([5, 1])
    with col_toggle:
        light = st.toggle("Modo claro", value=(mode == "light"), key="light_toggle")
    mode = "light" if light else "dark"
    st.session_state["theme_mode"] = mode
    inject_styles(mode)
    _inject_wizard_styles()

    with col_hdr:
        render_header(
            "Estimador de Precio de Viviendas",
            "Completa la información paso a paso para obtener una estimación precisa.",
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

    step = st.session_state["wizard_step"]
    errors: list[str] = st.session_state.get("wizard_errors", [])

    _render_indicator(step)
    _render_step_header(step)

    # ── Steps 0–3: data entry ────────────────────────────────────
    if step < len(WIZARD_STEPS):
        _render_step_fields(step, ref_df, defaults)

        if errors:
            items = "".join(f"• {e}<br>" for e in errors)
            st.markdown(
                f"<div class='wiz-err-box'>"
                f"<b>Por favor completa los siguientes campos antes de continuar:</b><br>{items}"
                f"Los campos marcados con ⚠ aparecen señalados arriba."
                f"</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        c_back, _, c_next = st.columns([2, 3, 2])
        with c_back:
            if step > 0 and st.button("← Anterior", use_container_width=True):
                _save_step(step)
                st.session_state.update(wizard_errors=[], wizard_step=step - 1)
                _restore_step(step - 1)
                st.rerun()
        with c_next:
            next_lbl = "Revisar resumen →" if step == len(WIZARD_STEPS) - 1 else "Siguiente →"
            if st.button(next_lbl, type="primary", use_container_width=True):
                missing = _validate_step(step)
                if missing:
                    st.session_state["wizard_errors"] = missing
                else:
                    _save_step(step)
                    st.session_state.update(wizard_errors=[], wizard_step=step + 1)
                    _restore_step(step + 1)
                st.rerun()

    # ── Step 4: summary ──────────────────────────────────────────
    elif step == STEP_SUMMARY:
        all_ready = _render_summary()
        st.write("")
        c_back, _, c_calc = st.columns([2, 3, 2])
        with c_back:
            if st.button("← Anterior", use_container_width=True):
                st.session_state.update(wizard_errors=[], wizard_step=len(WIZARD_STEPS) - 1)
                _restore_step(len(WIZARD_STEPS) - 1)
                st.rerun()
        with c_calc:
            btn_txt = "🏠 Calcular precio" if all_ready else "Completa los campos obligatorios *"
            if st.button(btn_txt, type="primary", disabled=not all_ready, use_container_width=True):
                result = compute_result(
                    _collect_values(), features, defaults, ref_df, metrics_df, linear_model, rf_model
                )
                st.session_state["result"] = result
                st.session_state.pop("explanation", None)
                st.session_state.pop("explanation_factors", None)
                st.session_state["wizard_step"] = STEP_RESULT
                st.rerun()

    # ── Step 5: result ───────────────────────────────────────────
    else:
        result = st.session_state.get("result")
        if result:
            render_result(result, metrics_df, rf_model)
        st.write("")
        st.divider()
        if st.button("← Nueva estimación"):
            _reset_wizard()
            st.rerun()


if __name__ == "__main__":
    main()
