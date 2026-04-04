"""Generate feature diagnostics inspired by COMPAÑERO analyses.

Outputs are written to reports/diagnostics:
- estadisticas_numericas.csv
- estadisticas_categoricas.csv
- correlacion_saleprice.csv
- importancia_rf.csv
- frecuencias_categoricas.csv
- baja_varianza.csv
- recomendaciones_features.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "diagnostics"
TARGET = "SalePrice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar diagnosticos de variables")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--low-info-threshold",
        type=float,
        default=0.85,
        help="Umbral de dominancia para marcar baja informacion",
    )
    return parser.parse_args()


def low_info_report(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in df.columns:
        vc = df[col].value_counts(normalize=True, dropna=False)
        if vc.empty:
            continue
        top_ratio = float(vc.iloc[0])
        if top_ratio >= threshold:
            top_value = vc.index[0]
            rows.append(
                {
                    "variable": col,
                    "top_value": str(top_value),
                    "top_ratio": top_ratio,
                    "n_unique": int(df[col].nunique(dropna=False)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["variable", "top_value", "top_ratio", "n_unique"])
    return pd.DataFrame(rows).sort_values(by="top_ratio", ascending=False)


def rf_importance_report(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns:
        return pd.DataFrame(columns=["variable", "importancia"])

    features = [c for c in df.columns if c not in {TARGET, "Id"}]
    x = df[features].select_dtypes(include=[np.number]).copy()
    if x.empty:
        return pd.DataFrame(columns=["variable", "importancia"])

    x = x.fillna(0)
    y = df[TARGET]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x, y)

    imp = pd.DataFrame(
        {"variable": x.columns, "importancia": model.feature_importances_}
    ).sort_values(by="importancia", ascending=False)
    return imp


def corr_report(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET not in df.columns:
        return pd.DataFrame(columns=["variable", "correlacion"])

    numeric = df.select_dtypes(include=[np.number])
    if TARGET not in numeric.columns:
        return pd.DataFrame(columns=["variable", "correlacion"])

    corr = numeric.corr(numeric_only=True)[TARGET].drop(labels=[TARGET], errors="ignore")
    corr_df = corr.abs().sort_values(ascending=False).rename("correlacion").reset_index()
    corr_df.columns = ["variable", "correlacion"]
    return corr_df


def categorical_frequency_report(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for col in df.select_dtypes(include=["object", "str"]).columns:
        freq = df[col].value_counts(dropna=False).reset_index()
        freq.columns = ["valor", "frecuencia"]
        freq["variable"] = col
        rows.append(freq[["variable", "valor", "frecuencia"]])
    if not rows:
        return pd.DataFrame(columns=["variable", "valor", "frecuencia"])
    return pd.concat(rows, ignore_index=True)


def recommendation_md(corr_df: pd.DataFrame, imp_df: pd.DataFrame, low_info_df: pd.DataFrame) -> str:
    corr_top = corr_df.head(20)
    imp_top = imp_df.head(20)

    corr_set = set(corr_top["variable"].tolist())
    imp_set = set(imp_top["variable"].tolist())
    low_info_set = set(low_info_df["variable"].tolist())

    suggested = sorted((corr_set | imp_set) - low_info_set)

    lines = [
        "# Recomendaciones de Variables",
        "",
        "Generado automaticamente con base en correlacion, importancia RF y columnas de baja informacion.",
        "",
        "## Top correlacion (absoluta) con SalePrice",
    ]
    for _, row in corr_top.iterrows():
        lines.append(f"- {row['variable']}: {row['correlacion']:.4f}")

    lines.append("")
    lines.append("## Top importancia Random Forest")
    for _, row in imp_top.iterrows():
        lines.append(f"- {row['variable']}: {row['importancia']:.4f}")

    lines.append("")
    lines.append("## Variables con baja informacion (>= umbral de dominancia)")
    if low_info_df.empty:
        lines.append("- Ninguna")
    else:
        for _, row in low_info_df.iterrows():
            lines.append(
                f"- {row['variable']}: top={row['top_value']} ratio={row['top_ratio']:.3f}"
            )

    lines.append("")
    lines.append("## Set sugerido (union de top correlacion + top importancia, excluyendo baja informacion)")
    if not suggested:
        lines.append("- Sin sugerencias")
    else:
        for col in suggested:
            lines.append(f"- {col}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.train_csv.exists():
        raise FileNotFoundError(f"No se encontro: {args.train_csv}")

    df = pd.read_csv(args.train_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats_num = df.describe(include=["number"]).T
    stats_cat = df.describe(include=["object", "str"]).T
    corr_df = corr_report(df)
    imp_df = rf_importance_report(df)
    freq_cat = categorical_frequency_report(df)
    low_info_df = low_info_report(df.drop(columns=[TARGET], errors="ignore"), args.low_info_threshold)

    stats_num.to_csv(args.out_dir / "estadisticas_numericas.csv", index=True)
    stats_cat.to_csv(args.out_dir / "estadisticas_categoricas.csv", index=True)
    corr_df.to_csv(args.out_dir / "correlacion_saleprice.csv", index=False)
    imp_df.to_csv(args.out_dir / "importancia_rf.csv", index=False)
    freq_cat.to_csv(args.out_dir / "frecuencias_categoricas.csv", index=False)
    low_info_df.to_csv(args.out_dir / "baja_varianza.csv", index=False)

    rec_text = recommendation_md(corr_df, imp_df, low_info_df)
    (args.out_dir / "recomendaciones_features.md").write_text(rec_text, encoding="utf-8")

    print(f"Diagnosticos generados en: {args.out_dir}")
    print(f"Top correlacion: {corr_df.head(5)['variable'].tolist()}")
    if not imp_df.empty:
        print(f"Top RF importance: {imp_df.head(5)['variable'].tolist()}")


if __name__ == "__main__":
    main()
