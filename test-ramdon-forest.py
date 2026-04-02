# Agente de IA, prediccion de precios de vivienda (Random Forest)

# --- FASE 1: LIBRERIAS Y CONFIGURACION ---
import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import utils

os.system("cls" if os.name == "nt" else "clear")

base_path = Path(__file__).parent
train_path = base_path / "output" / "train.csv"
test_path = base_path / "output" / "test.csv"

# --- FASE 2: CARGA DE DATOS Y SELECCION DE VARIABLES ---
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Variables seleccionadas y refinadas (ver justificación en variables_relevantes_modelo.md)
numeric_features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "GarageCars",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "FullBath",
    "TotRmsAbvGrd",
    "Fireplaces",
    "OverallCond",
]

categorical_features = [
    "Neighborhood",
    "ExterQual",
    "KitchenQual",
    "BsmtQual",
    "Foundation",
    "MSZoning",
    "SaleCondition",
    "CentralAir",
]

features = numeric_features + categorical_features

# --- FASE 3: PREPROCESAMIENTO DE DATOS ---
X = train[features].copy()
X_test_final = test[features].copy()

# Combinar para preprocesamiento consistente (mismas columnas dummy)
df_combined = pd.concat([X, X_test_final], ignore_index=True)

# Imputación de nulos (Mediana para numéricas, Moda para categóricas)
for col in numeric_features:
    df_combined[col] = df_combined[col].fillna(df_combined[col].median())

for col in categorical_features:
    df_combined[col] = df_combined[col].fillna(df_combined[col].mode()[0])

# Codificación One-Hot para variables categóricas
df_combined = pd.get_dummies(df_combined, columns=categorical_features, drop_first=True)

# Separar nuevamente en conjuntos de entrenamiento y prueba final
X_prep = df_combined.iloc[: len(train)].copy()
X_test_prep = df_combined.iloc[len(train) :].copy()

# --- FASE 4: TRANSFORMACION DEL TARGET ---
# Aplicar logaritmo iterativo (log1p) para normalizar la distribución de los precios
y = np.log1p(train["SalePrice"])

# --- FASE 5: DIVISIÓN DE ENTRENAMIENTO Y VALIDACIÓN ---
X_train, X_val, y_train, y_val = train_test_split(
    X_prep, y, test_size=0.2, random_state=42
)

# --- FASE 6: ENTRENAMIENTO DEL MODELO ---
print("⚙️ Entrenando el modelo Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=9, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)

# --- FASE 7: EVALUACION DEL MODELO ---
y_pred_val_log = rf_model.predict(X_val)

# Invertir logaritmo para obtener los valores en dólares (expm1)
y_val_exp = np.expm1(y_val)
y_pred_val_exp = np.expm1(y_pred_val_log)

rmse = np.sqrt(mean_squared_error(y_val_exp, y_pred_val_exp))
print(f"✅ Evaluación -> RMSE en Validación: ${rmse:,.2f}")

# --- FASE 8: PREDICCION FINAL Y GUARDADO ---
print("🚀 Generando predicciones sobre el set de prueba...")
y_pred_test_log = rf_model.predict(X_test_prep)
y_pred_test_exp = np.expm1(y_pred_test_log)

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": y_pred_test_exp})

submission_path = base_path / "output" / "submission_rf.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Predicciones guardadas exitosamente en: {submission_path}")
