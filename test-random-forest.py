# * Librerias/dependencias
import pandas as pd

# Cargar el dataset
train = pd.read_csv("output/train.csv")
test = pd.read_csv("output/test.csv")

# Variables seleccionadas para el modelo (ver justificación en variables_relevantes_modelo.md)
features = [
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "Neighborhood",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "GarageArea",
    "SaleType",
    "SaleCondition",
]

# Selección de variables para entrenamiento
X_train = train[features]
y_train = train["SalePrice"]
X_test = test[features]
