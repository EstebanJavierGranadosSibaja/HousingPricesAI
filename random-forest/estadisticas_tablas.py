from pathlib import Path
import shutil

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# Rutas de los archivos (ajusta si es necesario)

base_path = Path(__file__).parent
train_path = base_path / "output" / "train.csv"
test_path = base_path / "output" / "test.csv"

# Cargar los datos
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Estadísticas descriptivas
describe_num = df_train.describe(include=["number"]).T
describe_cat = df_train.describe(include=["object"]).T

# Correlación con SalePrice
if "SalePrice" in df_train.columns:
    corr = df_train.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
    corr = corr.drop("SalePrice", errors="ignore")
    corr = corr.to_frame(name="correlacion_SalePrice")
else:
    corr = pd.DataFrame()

corr["correlacion_SalePrice"] = corr["correlacion_SalePrice"].apply(
    lambda x: f"{x*100:.2f}%"
)

# Importancia de variables con Random Forest
importancia_rf = pd.DataFrame()
features_rf = [col for col in df_train.columns if col not in ["SalePrice", "Id"]]
df_rf = df_train[features_rf].copy()

# Preprocesamiento simple para Random Forest (solo variables numéricas, sin nulos)
df_rf_num = df_rf.select_dtypes(include=[np.number]).fillna(0)
if "SalePrice" in df_train.columns and not df_rf_num.empty:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(df_rf_num, df_train["SalePrice"])
    importancia_rf = pd.DataFrame(
        {"variable": df_rf_num.columns, "importancia": rf.feature_importances_}
    ).sort_values("importancia", ascending=False)

importancia_rf["importancia"] = importancia_rf["importancia"].apply(
    lambda x: f"{x*100:.2f}%"
)

# Unificar frecuencias de variables categóricas en un solo DataFrame
frecuencias_cat_list = []
for col in df_train.select_dtypes(include=["object"]).columns:
    freq = df_train[col].value_counts().reset_index()
    freq.columns = ["valor", "frecuencia"]
    freq["variable"] = col
    frecuencias_cat_list.append(freq)
if frecuencias_cat_list:
    frecuencias_cat_df = pd.concat(frecuencias_cat_list, ignore_index=True)[
        ["variable", "valor", "frecuencia"]
    ]
else:
    frecuencias_cat_df = pd.DataFrame(columns=["variable", "valor", "frecuencia"])

# Guardar las tablas en archivos CSV
output_dir = os.path.join(os.path.dirname(__file__), "output\\estadisticas")

shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir, exist_ok=True)

describe_num.to_csv(
    os.path.join(output_dir, "estadisticas_numericas.csv"), encoding="utf-8"
)
describe_cat.to_csv(
    os.path.join(output_dir, "estadisticas_categoricas.csv"), encoding="utf-8"
)
corr.to_csv(os.path.join(output_dir, "correlacion_SalePrice.csv"), encoding="utf-8")
if not importancia_rf.empty:
    importancia_rf.to_csv(
        os.path.join(output_dir, "importancia_rf.csv"), encoding="utf-8", index=False
    )


# Guardar frecuencias categóricas en un solo archivo
frecuencias_cat_df.to_csv(
    os.path.join(output_dir, "frecuencias_categoricas.csv"),
    encoding="utf-8",
    index=False,
)

print("Tablas de estadísticas guardadas en:")
print(os.path.join(output_dir, "estadisticas_numericas.csv"))
print(os.path.join(output_dir, "estadisticas_categoricas.csv"))
print(os.path.join(output_dir, "correlacion_SalePrice.csv"))
if not importancia_rf.empty:
    print(os.path.join(output_dir, "importancia_rf.csv"))
print(os.path.join(output_dir, "frecuencias_categoricas.csv"))
print("Frecuencias de variables categóricas guardadas en un solo archivo.")
