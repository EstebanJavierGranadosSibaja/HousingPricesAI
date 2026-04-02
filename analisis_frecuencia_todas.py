from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.system("cls" if os.name == "nt" else "clear")

# Rutas de los archivos (ajusta si es necesario)
# folder = "house-prices-advanced-regression-techniques\\"

base_path = Path(__file__).parent

train_path = base_path / "output" / "train.csv"
test_path = base_path / "output" / "test.csv"

# Cargar los datos
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Unir ambos datasets para análisis conjunto (sin la variable objetivo en test)
df_test["SalePrice"] = None  # Añadir columna vacía para igualar columnas
df_all = pd.concat([df_train, df_test], ignore_index=True)

# Crear carpeta de salida para las gráficas
output_dir = os.path.join(os.path.dirname(__file__), "output\\graficas_frecuencia")
os.makedirs(output_dir, exist_ok=True)

# Analizar todas las columnas
for variable in df_all.columns:
    plt.figure(figsize=(12, 6))
    # Si la variable es numérica y tiene muchos valores únicos, usar histograma
    if (
        pd.api.types.is_numeric_dtype(df_all[variable])
        and df_all[variable].nunique() > 20
    ):
        sns.histplot(x=df_all[variable].dropna(), kde=False, bins=30)
        plt.title(f"Histograma de {variable}")
    else:
        # Para variables categóricas o numéricas con pocos valores únicos
        orden = df_all[variable].value_counts().index
        sns.countplot(data=df_all, x=variable, order=orden)
        plt.title(f"Frecuencia de {variable}")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{variable}_frecuencia.png"))
    plt.close()

print(f"Gráficas guardadas en: {output_dir}")
