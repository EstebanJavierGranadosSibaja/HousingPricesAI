# ProyectoAI - Prediccion de Precio de Viviendas

Proyecto final de IA (regresion) para predecir precios de viviendas con dos modelos comparables:

- Regresion Lineal
- Random Forest

Incluye notebook principal, entrenamiento reproducible, evaluacion con metricas (MAE, MSE, RMSE, R2) y sistema interactivo con Streamlit.

## Estructura

```text
ProyectoAI/
├── data/
│   ├── raw/                 # train.csv, test.csv, sample_submission.csv
│   └── processed/           # datasets transformados
├── notebooks/
│   └── 01_housing_prices_proyecto_final.ipynb
├── src/
│   ├── preprocessing.py     # limpieza/transformacion
│   ├── train.py             # entrenamiento y evaluacion de ambos modelos
│   └── predict.py           # prediccion por CLI
├── app/
│   └── streamlit_app.py     # interfaz interactiva
├── models/                  # modelos entrenados (.joblib)
├── reports/
│   ├── metrics_comparison.csv
│   └── figures/
├── docs/
│   ├── EnunciadoProyecto.md
│   └── dataset/
├── scripts/
│   ├── setup_env.ps1
│   └── setup_env.sh
├── requirements.txt
└── README.md
```

## Setup rapido

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenamiento

```bash
python src/train.py
```

Salida esperada:

- Modelos en `models/linear_regression.joblib` y `models/random_forest.joblib`
- Metricas en `reports/metrics_comparison.csv`

## Prediccion por consola

```bash
python src/predict.py --tamano 120 --ubicacion NAmes --habitaciones 3
```

## Demo interactiva (sistema obligatorio)

```bash
streamlit run app/streamlit_app.py
```

## Flujo del proyecto

1. Ingesta de datos desde `data/raw/train.csv`.
2. Limpieza y transformacion (`src/preprocessing.py`).
3. Entrenamiento de Regresion Lineal y Random Forest (`src/train.py`).
4. Evaluacion comparativa con MAE, MSE, RMSE y R2.
5. Prediccion en CLI o Streamlit (`src/predict.py`, `app/streamlit_app.py`).
