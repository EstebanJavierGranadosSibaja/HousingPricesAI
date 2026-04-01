from pathlib import Path

import pandas as pd
import os

os.system("cls" if os.name == "nt" else "clear")

base_path = Path(__file__).parent
train_path = base_path / "output" / "train.csv"
test_path = base_path / "output" / "test.csv"


df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# ── Combinar para análisis estructural ───────────────────────────────────────
# SalePrice solo existe en train; en df_all quedará NaN para las filas de test
df_all = pd.concat([df_train, df_test], ignore_index=True)


# ── 1. BAJA VARIANZA: decidir qué columnas eliminar viendo ambos conjuntos ───
def get_cols_baja_varianza(df, threshold=0.85):
    cols_a_eliminar = []
    for col in df.columns:
        if col in ["Id", "SalePrice"]:
            continue
        top_freq = df[col].value_counts(normalize=True, dropna=False).max()
        if top_freq > threshold:
            cols_a_eliminar.append(col)
    return cols_a_eliminar


cols_a_eliminar = get_cols_baja_varianza(df_all)  # visión global
df_train = df_train.drop(columns=cols_a_eliminar, errors="ignore")
df_test = df_test.drop(columns=cols_a_eliminar, errors="ignore")
df_all = df_all.drop(columns=cols_a_eliminar, errors="ignore")


# ── 2. CATEGÓRICAS: categorías raras → "Otro" viendo ambos conjuntos ─────────
def agrupar_categorias_raras(df_all, df_train, df_test, threshold=0.01):
    df_train = df_train.copy()
    df_test = df_test.copy()
    cols_cat = df_all.select_dtypes(include=["object", "category"]).columns

    for col in cols_cat:
        freq = df_all[col].value_counts(normalize=True)  # frecuencia global
        categorias_validas = freq[freq >= threshold].index

        for df in [df_train, df_test]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: (
                        x if x in categorias_validas else ("Otro" if pd.notna(x) else x)
                    )
                )

    return df_train, df_test


df_train, df_test = agrupar_categorias_raras(df_all, df_train, df_test)


# ── 3. NUMÉRICAS: winsorizar — límites SOLO desde train ──────────────────────
# Aquí sí se usa solo train para no filtrar información del futuro al modelo
def winsorizar_numericas(df_train, df_test, lower=0.01, upper=0.99):
    df_train = df_train.copy()
    df_test = df_test.copy()
    cols_num = df_train.select_dtypes(include=["number"]).columns
    cols_num = [c for c in cols_num if c not in ["Id", "SalePrice"]]

    for col in cols_num:
        lo = df_train[col].quantile(lower)
        hi = df_train[col].quantile(upper)
        df_train[col] = df_train[col].clip(lo, hi)
        if col in df_test.columns:
            df_test[col] = df_test[col].clip(lo, hi)

    return df_train, df_test


df_train, df_test = winsorizar_numericas(df_train, df_test)

# ── Reporte ──────────────────────────────────────────────────────────────────
reporte = [
    "# Reporte de preprocesamiento\n",
    f"## Columnas eliminadas por baja varianza ({len(cols_a_eliminar)})\n",
    *[f"- {c}" for c in cols_a_eliminar],
    f"\n## Shape train: {df_train.shape}",
    f"## Shape test:  {df_test.shape}",
]
with open(
    os.path.join(output_dir, "reporte_preprocesamiento.md"), "w", encoding="utf-8"
) as f:
    f.write("\n".join(reporte))

# ── Guardar ──────────────────────────────────────────────────────────────────
df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False, na_rep="NA")
df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False, na_rep="NA")

print(f"Train: {df_train.shape}")
print(f"Test:  {df_test.shape}")
print(f"Columnas eliminadas: {cols_a_eliminar}")
print(f"Archivos guardados en: {output_dir}")
