# utilidades para el notebook
# * Librerias
# Manejo de datos
import numpy as np
import pandas as pd
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Conversión de variables categóricas a numéricas
from sklearn.preprocessing import LabelEncoder

# División de datos
from sklearn.model_selection import train_test_split

# Modelos
# from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Métricas de evaluación
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
from pathlib import Path

import shutil
from sklearn.ensemble import RandomForestRegressor


# ! funciones

# * base


def grab_col_names(dataframe, cat_th=1, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]

    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car, num_but_cat


# Copia para no modificar los originales
def encode_features(df):
    df = df.copy()
    # Label Encoding para variable binaria
    le = LabelEncoder()
    df["CentralAir"] = le.fit_transform(df["CentralAir"])  # N=0, Y=1

    # One-Hot Encoding para el resto de variables categóricas
    cat_vars = [
        "Neighborhood",
        "ExterQual",
        "KitchenQual",
        "BsmtQual",
        "Foundation",
        "MSZoning",
        "SaleCondition",
    ]
    df = pd.get_dummies(df, columns=cat_vars, drop_first=True)
    return df


def contar_columnas(df, detalle_tipos=False):
    """
    Retorna el número total de columnas del DataFrame.
    Si `detalle_tipos=True` devuelve un dict con totales y counts por tipo.
    """
    total = df.shape[1]
    if not detalle_tipos:
        return total

    num_numeric = df.select_dtypes(include=["number"]).shape[1]
    num_categorical = df.select_dtypes(exclude=["number"]).shape[1]
    return {"total": total, "numeric": num_numeric, "categorical": num_categorical}


# * Analisis de dataset
def diagrama_alta_correlacion(dataframe, mask_upper=True):
    df_num = dataframe.select_dtypes(include=["number"]).corr()
    n = df_num.shape[0]

    # tamaño y resolución: escala con n y dpi alto
    fig_w = max(8, n * 0.35)
    fig_h = max(6, n * 0.35)
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # máscara triángulo superior para evitar duplicados (opcional)
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(df_num, dtype=bool))

    sns.heatmap(df_num.corr(), mask=mask, annot=True, cmap="coolwarm")

    # etiquetas legibles
    plt.title("Matriz de alta correlación")
    plt.tight_layout()
    plt.show()


def columnas_alta_correlacion(dataframe, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)
    )
    drop_list = [
        col
        for col in upper_triangle_matrix.columns
        if any(upper_triangle_matrix[col] > corr_th)
    ]
    return drop_list


def high_correlated_cols(df, threshold=0.8, plot=False):
    # Korelasyon matrisini hesapla
    corr_matrix = df.corr()

    # Korelasyonu yüksek olan sütunları seç
    high_corr = []
    for col in corr_matrix.columns:
        for row in corr_matrix.index:
            if abs(corr_matrix.loc[row, col]) > threshold and row != col:
                high_corr.append((row, col, corr_matrix.loc[row, col]))

    # Yüksek korelasyonlu çiftler
    high_corr = list(set(high_corr))  # Aynı çiftin birden fazla görünmesini engelle

    if plot:
        # Korelasyon ısı haritası çiz
        plt.figure(figsize=(30, 30))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Korelasyon Isı Haritası")
        plt.show()

    return high_corr


# *  Estadisticas y reportes


def graficos_ocurrencia_variables(dataframe):
    # Crear carpeta de salida para las gráficas
    base_dir = Path(__file__).parent
    output_dir = base_dir / "output" / "graficas_frecuencia"

    os.makedirs(output_dir, exist_ok=True)

    # Analizar todas las columnas
    for variable in dataframe.columns:
        plt.figure(figsize=(12, 6))
        # Si la variable es numérica y tiene muchos valores únicos, usar histograma
        if (
            pd.api.types.is_numeric_dtype(dataframe[variable])
            and dataframe[variable].nunique() > 20
        ):
            sns.histplot(x=dataframe[variable].dropna(), kde=False, bins=30)
            plt.title(f"Histograma de {variable}")
        else:
            # Para variables categóricas o numéricas con pocos valores únicos
            orden = dataframe[variable].value_counts().index
            sns.countplot(data=dataframe, x=variable, order=orden)
            plt.title(f"Frecuencia de {variable}")
            plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{variable}_frecuencia.png"))
        plt.close()

    print(f"Gráficas guardadas en: {output_dir}")


def reporte_variables(dataframe):
    base_dir = Path(__file__).parent
    reporte = "# Variables obtenidas\n\n"

    for col in dataframe.columns:
        valores = dataframe[col].dropna().unique()
        tipo = dataframe[col].dtype
        reporte += f"## {col}\n\n"
        if tipo in ["float64", "int64"]:
            if len(valores) > 15:
                reporte += "[Datos numéricos variables]\n\n"
            else:
                valores_ordenados = sorted(valores)
                reporte += ", ".join(str(v) for v in valores_ordenados) + "\n\n"
        else:
            valores_ordenados = sorted(
                valores, key=lambda x: (str(x).lower() if isinstance(x, str) else x)
            )
            reporte += ", ".join(str(v) for v in valores_ordenados) + "\n\n"

    with open(base_dir / "output" / "reporte_variables.md", "w", encoding="utf-8") as f:
        f.write(reporte)


def estadisticas(dataset_path=None):
    if dataset_path == None:
        dataset_path = Path(__file__).parent
    train_path = dataset_path / "output" / "train.csv"
    test_path = dataset_path / "output" / "test.csv"

    # Cargar los datos
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Estadísticas descriptivas
    describe_num = df_train.describe(include=["number"]).T
    describe_cat = df_train.describe(include=["object"]).T

    # Correlación con SalePrice
    if "SalePrice" in df_train.columns:
        corr = df_train.corr(numeric_only=True)["SalePrice"].sort_values(
            ascending=False
        )
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
            os.path.join(output_dir, "importancia_rf.csv"),
            encoding="utf-8",
            index=False,
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
