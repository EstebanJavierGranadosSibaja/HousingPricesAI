"""Premium dual-theme (dark / light) UI helpers for the Streamlit housing app."""

from __future__ import annotations

import streamlit as st

# Palette per theme. Everything visual is driven by these CSS variables so the
# light/dark toggle switches the whole interface cleanly.
_PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "app_bg": "radial-gradient(1100px 460px at 50% -8%, rgba(226,178,87,0.12), transparent 60%), "
                  "linear-gradient(180deg,#0d0e13 0%,#0b0c10 45%,#08090c 100%)",
        "ink": "#f3f4f6",
        "muted": "#9aa3b2",
        "line": "rgba(255,255,255,0.08)",
        "panel": "linear-gradient(180deg,#15171f 0%,#121420 100%)",
        "input_bg": "#1c1f2a",
        "input_border": "rgba(255,255,255,0.12)",
        "track": "rgba(255,255,255,0.08)",
        "hero": "linear-gradient(135deg,#1b1e27 0%,#101218 100%)",
        "hero_border": "rgba(226,178,87,0.35)",
        "hero_glow": "0 24px 60px rgba(0,0,0,0.55), inset 0 0 60px rgba(226,178,87,0.05)",
        "title_grad": "linear-gradient(90deg,#ffffff,#f0d8a8)",
        "price_grad": "linear-gradient(90deg,#ffffff,#f0d8a8 70%)",
        "label": "#dfe3ea",
        "accent": "#e2b257",
        "accent2": "#c9923b",
        "menu_bg": "#1c1f2a",
        "shadow": "0 20px 50px rgba(0,0,0,0.45)",
    },
    "light": {
        "app_bg": "radial-gradient(1100px 460px at 50% -8%, rgba(201,146,59,0.10), transparent 60%), "
                  "linear-gradient(180deg,#fcfcfe 0%,#f4f5f8 50%,#eef0f5 100%)",
        "ink": "#10141d",
        "muted": "#5b6472",
        "line": "#e6e8ef",
        "panel": "linear-gradient(180deg,#ffffff 0%,#fbfcfe 100%)",
        "input_bg": "#ffffff",
        "input_border": "#dfe3ec",
        "track": "#ebedf3",
        "hero": "linear-gradient(135deg,#fffaf0 0%,#fdf6e7 100%)",
        "hero_border": "rgba(184,134,11,0.45)",
        "hero_glow": "0 22px 48px rgba(20,30,60,0.12)",
        "title_grad": "linear-gradient(90deg,#10141d,#9a7321)",
        "price_grad": "linear-gradient(90deg,#2a2410,#9a7321 85%)",
        "label": "#39404d",
        "accent": "#b8860b",
        "accent2": "#9a7321",
        "menu_bg": "#ffffff",
        "shadow": "0 18px 40px rgba(20,30,60,0.10)",
    },
}


def inject_styles(mode: str = "dark") -> None:
    """Inject the premium theme (dark by default, or light)."""
    p = _PALETTES.get(mode, _PALETTES["dark"])
    css_vars = "".join(f"--{k}: {v};" for k, v in p.items())
    st.markdown(
        f"""
        <style>
        :root {{ {css_vars} }}
        .stApp {{ background: var(--app_bg); color: var(--ink); }}
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li, .stApp div {{ color: var(--ink); }}
        .main .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1160px; }}
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Header */
        .app-title {{
            font-size: 2.05rem; font-weight: 800; letter-spacing: -0.02em; margin: 0;
            background: var(--title_grad); -webkit-background-clip: text;
            -webkit-text-fill-color: transparent; background-clip: text;
        }}
        .app-subtitle {{ color: var(--muted) !important; font-size: 0.98rem; margin: 0.3rem 0 0 0; }}

        /* Sections */
        .section-title {{ font-size: 1.06rem; font-weight: 700; color: var(--ink); margin: 0.2rem 0 0.1rem 0; }}
        .section-subtitle {{ color: var(--muted) !important; font-size: 0.86rem; margin-bottom: 0.7rem; }}

        /* Input form */
        [data-testid="stForm"] {{
            background: var(--panel); border: 1px solid var(--line);
            border-radius: 18px; padding: 1.1rem 1.2rem 0.7rem 1.2rem; box-shadow: var(--shadow);
        }}
        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {{
            color: var(--label) !important; font-weight: 600; font-size: 0.9rem;
        }}
        [data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}

        /* Inputs / selects / number inputs follow the theme */
        [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div {{
            background: var(--input_bg) !important; border-color: var(--input_border) !important; border-radius: 10px !important;
        }}
        [data-baseweb="input"] input, [data-baseweb="select"] div, [data-testid="stNumberInput"] input {{
            color: var(--ink) !important;
        }}
        [data-testid="stNumberInput"] button {{ background: var(--input_bg) !important; color: var(--ink) !important; border-color: var(--input_border) !important; }}
        [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {{ background: var(--menu_bg) !important; }}
        [data-baseweb="popover"] [role="option"] {{ color: var(--ink) !important; }}

        /* Tabs */
        [data-baseweb="tab-list"] {{ gap: 0.25rem; border-bottom: 1px solid var(--line); }}
        [data-baseweb="tab"] {{ font-weight: 600; color: var(--muted) !important; }}
        [data-baseweb="tab"][aria-selected="true"] {{ color: var(--accent) !important; }}
        [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; }}

        /* Theme toggle (top-right) */
        [data-testid="stToggle"] {{ justify-content: flex-end; }}

        /* Price hero */
        .hero {{
            border-radius: 22px; padding: 1.8rem 2rem; background: var(--hero);
            border: 1px solid var(--hero_border); box-shadow: var(--hero_glow);
        }}
        .hero .label {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.16em; color: var(--accent); margin-bottom: 0.35rem; font-weight: 700; }}
        .hero .price {{
            font-size: 3.2rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em;
            background: var(--price_grad); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }}
        .hero .range {{ margin-top: 0.8rem; font-size: 0.97rem; color: var(--muted) !important; }}
        .hero .range b {{ color: var(--ink) !important; }}
        .hero .leader {{
            margin-top: 0.65rem; display: inline-block; background: color-mix(in srgb, var(--accent) 14%, transparent);
            border: 1px solid var(--hero_border); color: var(--accent) !important;
            padding: 0.32rem 0.8rem; border-radius: 999px; font-size: 0.82rem; font-weight: 700;
        }}

        /* Empty state */
        .empty {{ border: 1px dashed var(--line); border-radius: 18px; padding: 2.6rem 1.5rem; text-align: center; background: var(--panel); }}
        .empty .big {{ font-size: 1.06rem; font-weight: 700; color: var(--ink); margin-bottom: 0.3rem; }}
        .empty .sub {{ color: var(--muted) !important; }}

        /* Metric cards */
        [data-testid="stMetric"] {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 0.9rem 1rem; box-shadow: var(--shadow); }}
        [data-testid="stMetricLabel"] p {{ color: var(--muted) !important; font-weight: 600; }}
        [data-testid="stMetricValue"] {{ color: var(--ink) !important; font-weight: 800; }}

        /* Custom bar visualizations */
        .bars {{ margin-top: 0.2rem; }}
        .bar-row {{ margin-bottom: 0.7rem; }}
        .bar-top {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.3rem; }}
        .bar-top .l {{ font-size: 0.88rem; color: var(--ink); font-weight: 600; }}
        .bar-top .v {{ font-size: 0.85rem; color: var(--muted) !important; font-variant-numeric: tabular-nums; }}
        .bar-track {{ height: 13px; border-radius: 999px; background: var(--track); overflow: hidden; }}
        .bar-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent2), var(--accent)); }}
        .bar-fill.strong {{ box-shadow: 0 0 16px color-mix(in srgb, var(--accent) 55%, transparent); }}

        /* Custom performance table */
        .ptable {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
        .ptable th {{ text-align: left; color: var(--muted) !important; font-weight: 600; padding: 0.55rem 0.6rem; border-bottom: 1px solid var(--line); }}
        .ptable td {{ padding: 0.6rem 0.6rem; border-bottom: 1px solid var(--line); color: var(--ink); font-variant-numeric: tabular-nums; }}
        .ptable tr.best td {{ font-weight: 700; }}
        .ptable .badge {{ background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent) !important; padding: 0.12rem 0.5rem; border-radius: 999px; font-size: 0.74rem; font-weight: 700; margin-left: 0.4rem; }}

        /* Buttons */
        .stButton > button, .stFormSubmitButton > button {{
            width: 100%; border-radius: 12px; border: 0; padding: 0.68rem 1rem;
            background: linear-gradient(90deg, var(--accent), var(--accent2));
            color: #1a1407 !important; font-weight: 800; letter-spacing: 0.01em;
            box-shadow: 0 12px 28px color-mix(in srgb, var(--accent) 28%, transparent);
        }}
        .stButton > button:hover, .stFormSubmitButton > button:hover {{ filter: brightness(1.06); }}
        .stButton > button p, .stFormSubmitButton > button p {{ color: #1a1407 !important; }}

        [data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }}
        .stAlert {{ border-radius: 14px; }}
        hr {{ border-color: var(--line); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"<h1 class='app-title'>{title}</h1><p class='app-subtitle'>{subtitle}</p>",
        unsafe_allow_html=True,
    )


def render_section(title: str, subtitle: str = "") -> None:
    block = f"<div class='section-title'>{title}</div>"
    if subtitle:
        block += f"<div class='section-subtitle'>{subtitle}</div>"
    st.markdown(block, unsafe_allow_html=True)


def render_hero(price: float, low: float, high: float, leader_text: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="label">Precio estimado</div>
            <div class="price">${price:,.0f}</div>
            <div class="range">Rango probable: <b>${low:,.0f}</b> &nbsp;–&nbsp; <b>${high:,.0f}</b></div>
            <div class="leader">{leader_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty">
            <div class="big">Completa los datos de la vivienda y pulsa Calcular precio</div>
            <div class="sub">Verás la estimación con su rango, la comparación entre modelos
            y las variables más influyentes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bars(rows: list[dict]) -> None:
    """Render a premium horizontal-bar list. Each row: {label, value, frac, strong}."""
    html = "<div class='bars'>"
    for r in rows:
        frac = max(2.0, min(100.0, float(r.get("frac", 0.0)) * 100.0))
        strong = " strong" if r.get("strong") else ""
        html += (
            "<div class='bar-row'>"
            f"<div class='bar-top'><span class='l'>{r['label']}</span><span class='v'>{r['value']}</span></div>"
            f"<div class='bar-track'><div class='bar-fill{strong}' style='width:{frac:.1f}%'></div></div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_table(headers: list[str], rows: list[list[str]], best_row: int | None = None) -> None:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for i, row in enumerate(rows):
        cls = " class='best'" if best_row is not None and i == best_row else ""
        cells = "".join(f"<td>{c}</td>" for c in row)
        body += f"<tr{cls}>{cells}</tr>"
    st.markdown(f"<table class='ptable'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", unsafe_allow_html=True)
