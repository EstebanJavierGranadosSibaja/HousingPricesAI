"""Configuration constants for the Streamlit housing app."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics_comparison.csv"

PRIMARY_INPUTS = ["GrLivArea", "Neighborhood", "BedroomAbvGr"]

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
        "help": "Puntaje 1-10 estimado con reglas basadas en materiales, acabados y estado.",
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

OVERALL_CONDITION_CHOICES: list[tuple[str, int]] = [
    ("Necesita reparaciones importantes", 2),
    ("Estado regular con desgaste visible", 4),
    ("Estado funcional y estable", 6),
    ("Buena conservacion", 8),
    ("Excelente conservacion", 9),
]

QUALITY_MATERIAL_FACTS: dict[str, int] = {
    "Madera basica o prefabricado": 2,
    "Concreto y bloque estandar": 4,
    "Ladrillo o piedra de buena calidad": 6,
    "Metal estructural con acabados premium": 8,
    "Marmol o piedra de lujo": 9,
}

QUALITY_FINISH_FACTS: dict[str, int] = {
    "Acabados basicos": 1,
    "Acabados estandar": 2,
    "Acabados buenos": 3,
    "Acabados premium": 4,
    "Acabados de lujo": 5,
}

QUALITY_CONDITION_FACTS: dict[str, int] = {
    "Necesita remodelacion mayor": 1,
    "Estado regular": 2,
    "Estado funcional": 3,
    "Muy bien mantenida": 4,
    "Estado excelente": 5,
}

QUALITY_CODE_BY_LEVEL = {
    1: "Po",
    2: "Fa",
    3: "TA",
    4: "Gd",
    5: "Ex",
}

QUALITY_LEVEL_BY_CODE = {
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5,
}

QUALITY_CATEGORY_RULESETS: dict[str, dict[str, object]] = {
    "KitchenQual": {
        "intro": "Calidad de cocina segun materiales, acabados y estado.",
        "material_question": "1) Material principal de encimeras/muebles",
        "material_facts": {
            "Laminado basico": 1,
            "Madera estandar": 2,
            "Granito o cuarzo": 4,
            "Marmol o piedra premium": 5,
        },
        "finish_question": "2) Nivel de equipamiento y acabados",
        "finish_facts": {
            "Basico": 1,
            "Estandar": 2,
            "Bueno": 3,
            "Premium": 4,
            "Lujo": 5,
        },
        "condition_question": "3) Estado actual de la cocina",
        "condition_facts": {
            "Desgastada": 1,
            "Funcional": 2,
            "Buena": 3,
            "Muy buena": 4,
            "Excelente": 5,
        },
    },
    "ExterQual": {
        "intro": "Calidad exterior segun material de fachada y estado.",
        "material_question": "1) Material predominante de fachada",
        "material_facts": {
            "Revestimiento economico": 1,
            "Bloque o concreto estandar": 2,
            "Ladrillo": 3,
            "Piedra de buena calidad": 4,
            "Piedra premium o marmol": 5,
        },
        "finish_question": "2) Nivel de acabado exterior",
        "finish_facts": {
            "Basico": 1,
            "Estandar": 2,
            "Bueno": 3,
            "Premium": 4,
            "Lujo": 5,
        },
        "condition_question": "3) Estado de fachada y pintura",
        "condition_facts": {
            "Deteriorado": 1,
            "Regular": 2,
            "Bueno": 3,
            "Muy bueno": 4,
            "Excelente": 5,
        },
    },
    "BsmtQual": {
        "intro": "Calidad de sotano segun estructura, acabados y estado.",
        "material_question": "1) Estructura y material predominante del sotano",
        "material_facts": {
            "No aplica (sin sotano)": 0,
            "Concreto basico": 2,
            "Concreto reforzado": 3,
            "Concreto reforzado premium": 5,
        },
        "finish_question": "2) Nivel de acabado del sotano",
        "finish_facts": {
            "No aplica (sin sotano)": 0,
            "Sin terminar": 1,
            "Semi terminado": 2,
            "Terminado funcional": 3,
            "Terminado premium": 5,
        },
        "condition_question": "3) Estado actual del sotano",
        "condition_facts": {
            "Sin sotano": 0,
            "Con humedad o deterioro": 1,
            "Regular": 2,
            "Bueno": 3,
            "Muy bueno": 4,
            "Excelente": 5,
        },
        "none_condition": "Sin sotano",
    },
}
