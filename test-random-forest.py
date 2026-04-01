#
# * Librerias
# Manejo de datos
import pandas as pd
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# División de datos
from sklearn.model_selection import train_test_split

# Modelos
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Métricas de evaluación
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
from pathlib import Path

os.system("cls" if os.name == "nt" else "clear")

# * Cargar el dataset
base_path = Path(__file__).parent

train_path = base_path / "output" / "train.csv"
test_path = base_path / "output" / "test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Variables seleccionadas para el modelo (ver justificación en variables_relevantes_modelo.md)
features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "BsmtFinSF1",
    "1stFlrSF",
    "GarageCars",
    "LotArea",
    "GarageArea",
    "YearRemodAdd",
    "YearBuilt",
    "FullBath",
    "OpenPorchSF",
    "2ndFlrSF",
    "LotFrontage",
    "GarageYrBlt",
    "TotRmsAbvGrd",
    "BsmtUnfSF",
    "WoodDeckSF",
    "MoSold",
    "OverallCond",
    # Variables categóricas importantes que se deben codificar
    "Neighborhood",
    "MSZoning",
    "KitchenQual",
    "SaleCondition",
    "SaleType",
]

# * Selección de variables para entrenamiento
X_train = train[features]
y_train = train["SalePrice"]

X_test = test[features]
