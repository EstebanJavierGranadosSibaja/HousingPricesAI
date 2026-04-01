import pandas as pd
import os

os.system("cls" if os.name == "nt" else "clear")

train_path = os.path.join(
    os.path.dirname(__file__), "house-prices-advanced-regression-techniques\\train.csv"
)
test_path = os.path.join(
    os.path.dirname(__file__), "house-prices-advanced-regression-techniques\\test.csv"
)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)


# ── 1. CATEGÓRICAS: agrupar categorías raras en "Otro" ───────────────────────
# Se aprende qué categorías son raras SOLO en train, luego se aplica a ambos.
def agrupar_categorias_raras(df_train, df_test, threshold=0.01):
    df_train = df_train.copy()
    df_test = df_test.copy()
    cols_categoricas = df_train.select_dtypes(include=["object", "category"]).columns

    for col in cols_categoricas:
        freq = df_train[col].value_counts(normalize=True)
        categorias_validas = freq[freq >= threshold].index

        df_train[col] = df_train[col].apply(
            lambda x: x if x in categorias_validas else ("Otro" if pd.notna(x) else x)
        )
        df_test[col] = df_test[col].apply(
            lambda x: x if x in categorias_validas else ("Otro" if pd.notna(x) else x)
        )

    return df_train, df_test


# ── 2. COLUMNAS DE BAJA VARIANZA: eliminar si >85% es un solo valor ──────────
# Se decide qué columnas eliminar SOLO mirando train, luego se aplica a ambos.
def eliminar_baja_varianza(df_train, df_test, threshold=0.85):
    cols_a_eliminar = []
    for col in df_train.columns:
        if col in ["Id", "SalePrice"]:  # nunca tocar target ni ID
            continue
        top_freq = df_train[col].value_counts(normalize=True, dropna=False).max()
        if top_freq > threshold:
            cols_a_eliminar.append(col)

    df_train = df_train.drop(columns=cols_a_eliminar)
    df_test = df_test.drop(columns=[c for c in cols_a_eliminar if c in df_test.columns])

    return df_train, df_test, cols_a_eliminar


# ── 3. NUMÉRICAS: winsorizar outliers extremos ───────────────────────────────
# Recorta valores por encima/debajo del percentil 1%-99% según train.
# Esto evita que valores rarísimos distorsionen el modelo sin eliminar filas.
def winsorizar_numericas(df_train, df_test, lower=0.01, upper=0.99):
    df_train = df_train.copy()
    df_test = df_test.copy()
    cols_numericas = df_train.select_dtypes(include=["number"]).columns
    cols_numericas = [c for c in cols_numericas if c not in ["Id", "SalePrice"]]

    limites = {}
    for col in cols_numericas:
        lo = df_train[col].quantile(lower)
        hi = df_train[col].quantile(upper)
        limites[col] = (lo, hi)
        df_train[col] = df_train[col].clip(lo, hi)

    for col in cols_numericas:
        if col in df_test.columns:
            lo, hi = limites[col]
            df_test[col] = df_test[col].clip(lo, hi)

    return df_train, df_test


# ── Aplicar en orden ─────────────────────────────────────────────────────────
df_train_f, df_test_f = agrupar_categorias_raras(df_train, df_test, threshold=0.01)
df_train_f, df_test_f, cols_eliminadas = eliminar_baja_varianza(
    df_train_f, df_test_f, threshold=0.85
)
df_train_f, df_test_f = winsorizar_numericas(df_train_f, df_test_f)

# ── Reporte ──────────────────────────────────────────────────────────────────
reporte = []
reporte.append(f"# Reporte de preprocesamiento\n")
reporte.append(f"## Columnas eliminadas por baja varianza ({len(cols_eliminadas)})\n")
for c in cols_eliminadas:
    reporte.append(f"- {c}")
reporte.append(f"\n## Shape train: {df_train_f.shape}")
reporte.append(f"## Shape test:  {df_test_f.shape}")

with open(
    os.path.join(output_dir, "reporte_preprocesamiento.md"), "w", encoding="utf-8"
) as f:
    f.write("\n".join(reporte))

# ── Guardar ──────────────────────────────────────────────────────────────────
df_train_f.to_csv(os.path.join(output_dir, "train.csv"), index=False, na_rep="NA")
df_test_f.to_csv(os.path.join(output_dir, "test.csv"), index=False, na_rep="NA")

print(f"Train: {df_train.shape} → {df_train_f.shape}")
print(f"Test:  {df_test.shape}  → {df_test_f.shape}")
print(f"Columnas eliminadas: {cols_eliminadas}")
print(f"Archivos guardados en: {output_dir}")
