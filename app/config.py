"""Configuration constants for the Streamlit housing app."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_comparison.csv"

INTEGER_LIKE_FEATURES = {
    "YearBuilt",
    "YearRemodAdd",
    "OverallQual",
    "OverallCond",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageCars",
    "Fireplaces",
}

NUMERIC_FEATURE_HINTS = {
    "GrLivArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "OverallQual",
    "OverallCond",
    "GarageCars",
    "GarageArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
}

FEATURE_GROUP_BLUEPRINT = [
    (
        "Ubicacion y mercado",
        ["MSZoning", "Neighborhood", "LotArea"],
    ),
    (
        "Tamano y distribucion",
        [
            "GrLivArea",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
        ],
    ),
    (
        "Calidad y acabados",
        ["OverallQual", "OverallCond", "KitchenQual", "BsmtQual", "ExterQual"],
    ),
    (
        "Estructura y garage",
        ["GarageCars", "GarageArea", "Foundation"],
    ),
    (
        "Tiempo",
        ["YearBuilt", "YearRemodAdd"],
    ),
]

FEATURE_METADATA: dict[str, dict[str, str]] = {
    "MSZoning": {
        "label": "Tipo de zonificacion municipal",
        "help": "Define el tipo de uso permitido para el lote (residencial, comercial, etc.).",
    },
    "Neighborhood": {
        "label": "Barrio o zona",
        "help": "Ubicacion general de la vivienda dentro de la ciudad.",
    },
    "LotArea": {
        "label": "Tamano total del lote (ft2)",
        "help": "Superficie total del terreno donde esta construida la vivienda.",
    },
    "LotFrontage": {
        "label": "Frente del lote (pies lineales)",
        "help": "Longitud del frente del lote que da a la calle.",
    },
    "YearBuilt": {
        "label": "Anio de construccion",
        "help": "Anio en que se construyo originalmente la vivienda.",
    },
    "YearRemodAdd": {
        "label": "Anio de ultima remodelacion",
        "help": "Anio de la remodelacion o renovacion mas reciente.",
    },
    "OverallQual": {
        "label": "Calidad general de materiales y acabados",
        "help": "Puntaje de 1 (muy basica) a 10 (lujo) de materiales y acabados.",
    },
    "OverallCond": {
        "label": "Estado general de conservacion",
        "help": "Nivel general de mantenimiento y desgaste de la vivienda.",
    },
    "GrLivArea": {
        "label": "Area habitable sobre nivel del suelo (ft2)",
        "help": "Area util interior en pisos principales, sin contar sotano.",
    },
    "1stFlrSF": {
        "label": "Area del primer piso (ft2)",
        "help": "Metros de superficie del primer piso en pies cuadrados.",
    },
    "2ndFlrSF": {
        "label": "Area del segundo piso (ft2)",
        "help": "Metros de superficie del segundo piso en pies cuadrados.",
    },
    "TotalBsmtSF": {
        "label": "Area total del sotano (ft2)",
        "help": "Superficie completa del sotano, terminado o no.",
    },
    "BsmtFinSF1": {
        "label": "Area de sotano terminada (ft2)",
        "help": "Parte del sotano que ya tiene acabado utilizable.",
    },
    "BsmtUnfSF": {
        "label": "Area de sotano sin terminar (ft2)",
        "help": "Parte del sotano en obra gris o sin acabados.",
    },
    "FullBath": {
        "label": "Baños completos",
        "help": "Cantidad de baños completos (lavamanos, sanitario y ducha/banera).",
    },
    "HalfBath": {
        "label": "Medios baños",
        "help": "Cantidad de baños con sanitario y lavamanos.",
    },
    "BedroomAbvGr": {
        "label": "Dormitorios sobre nivel del suelo",
        "help": "Numero de dormitorios en pisos principales.",
    },
    "KitchenQual": {
        "label": "Calidad de cocina",
        "help": "Calidad percibida de acabados y equipamiento de cocina.",
    },
    "TotRmsAbvGrd": {
        "label": "Total de habitaciones sobre nivel del suelo",
        "help": "Cantidad total de cuartos habitables, excluyendo baños.",
    },
    "GarageCars": {
        "label": "Capacidad de autos en garage",
        "help": "Numero aproximado de autos que caben en el garage.",
    },
    "GarageArea": {
        "label": "Area de garage (ft2)",
        "help": "Superficie del garage en pies cuadrados.",
    },
    "GarageYrBlt": {
        "label": "Anio de construccion del garage",
        "help": "Anio en que se construyo el garage.",
    },
    "GarageFinish": {
        "label": "Nivel de acabado del garage",
        "help": "Indica si el garage esta terminado, semi terminado o sin terminar.",
    },
    "MasVnrArea": {
        "label": "Area de revestimiento de mamposteria (ft2)",
        "help": "Superficie de revestimiento decorativo exterior en mamposteria.",
    },
    "MasVnrType": {
        "label": "Tipo de revestimiento exterior",
        "help": "Material principal del revestimiento decorativo de fachada.",
    },
    "OpenPorchSF": {
        "label": "Area de porche abierto (ft2)",
        "help": "Superficie de porche sin cerramiento.",
    },
    "WoodDeckSF": {
        "label": "Area de terraza de madera (ft2)",
        "help": "Superficie de deck o terraza de madera.",
    },
    "Fireplaces": {
        "label": "Cantidad de chimeneas",
        "help": "Numero de chimeneas en la vivienda.",
    },
    "BsmtQual": {
        "label": "Calidad del sotano",
        "help": "Calidad general percibida del sotano.",
    },
    "SaleCondition": {
        "label": "Condicion de venta",
        "help": "Contexto de la venta, por ejemplo normal o urgente.",
    },
    "ExterQual": {
        "label": "Calidad del exterior",
        "help": "Calidad de los materiales y acabados exteriores.",
    },
    "Foundation": {
        "label": "Tipo de cimientos",
        "help": "Sistema estructural de base usado en la vivienda.",
    },
}

CATEGORICAL_OPTION_LABELS_BY_FEATURE: dict[str, dict[str, str]] = {
    "ExterQual": {
        "Ex": "Excelente",
        "Gd": "Buena",
        "TA": "Promedio",
        "Fa": "Regular",
        "Po": "Mala",
        "Missing": "Sin dato",
    },
    "KitchenQual": {
        "Ex": "Excelente",
        "Gd": "Buena",
        "TA": "Promedio",
        "Fa": "Regular",
        "Po": "Mala",
        "Missing": "Sin dato",
    },
    "BsmtQual": {
        "Ex": "Excelente",
        "Gd": "Buena",
        "TA": "Promedio",
        "Fa": "Regular",
        "Po": "Mala",
        "NoBasement": "Sin sotano",
        "Missing": "Sin dato",
    },
    "GarageFinish": {
        "Fin": "Terminado",
        "RFn": "Semi terminado",
        "Unf": "Sin terminar",
        "NoGarage": "Sin garage",
        "Missing": "Sin dato",
    },
    "MasVnrType": {
        "BrkFace": "Ladrillo (frente)",
        "BrkCmn": "Ladrillo comun",
        "Stone": "Piedra",
        "None": "Sin revestimiento",
        "NoMasonry": "Sin mamposteria",
        "Missing": "Sin dato",
    },
    "SaleCondition": {
        "Normal": "Venta normal",
        "Abnorml": "Venta atipica",
        "AdjLand": "Ajuste por terreno",
        "Alloca": "Asignacion entre propiedades",
        "Family": "Venta entre familiares",
        "Partial": "Venta parcial",
        "Missing": "Sin dato",
    },
    "Foundation": {
        "BrkTil": "Ladrillo y baldosa",
        "CBlock": "Bloque de concreto",
        "PConc": "Concreto reforzado",
        "Slab": "Losa",
        "Stone": "Piedra",
        "Wood": "Madera",
        "Missing": "Sin dato",
    },
    "MSZoning": {
        "A": "Agricola",
        "C": "Comercial",
        "FV": "Residencial flotante",
        "I": "Industrial",
        "RH": "Residencial alta densidad",
        "RL": "Residencial baja densidad",
        "RM": "Residencial media densidad",
        "Missing": "Sin dato",
    },
}
