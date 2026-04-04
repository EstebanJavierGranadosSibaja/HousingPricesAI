# ProyectoAI - Prediccion de Precio de Viviendas

Proyecto final de IA (regresion) para predecir precios de viviendas con dos modelos comparables:

- Regresion Lineal
- Random Forest

Incluye notebook principal, entrenamiento reproducible, evaluacion con metricas (MAE, MSE, RMSE, R2) y sistema interactivo con Streamlit.

## Dependencias

- `requirements.txt`: dependencias minimas para ejecutar entrenamiento, prediccion y app.
- `requirements-notebook.txt`: dependencias opcionales para notebooks/experimentacion.

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
│   ├── feature_diagnostics.py
│   ├── setup_env.ps1
│   ├── run_app.ps1
│   ├── setup_env.sh
│   └── run_app.sh
├── requirements.txt
└── README.md
```

## Setup rapido

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Si vas a ejecutar notebooks con librerias adicionales:

```bash
pip install -r requirements-notebook.txt
```

## Setup recomendado para entrega (profesor)

Windows PowerShell:

```powershell
.\scripts\setup_env.ps1
.\scripts\run_app.ps1
```

Si quieres definir puerto/headless:

```powershell
.\scripts\run_app.ps1 --server.port 8507 --server.headless true
```

Linux/macOS:

```bash
chmod +x scripts/setup_env.sh scripts/run_app.sh
./scripts/setup_env.sh
./scripts/run_app.sh
```

Con argumentos opcionales:

```bash
./scripts/run_app.sh --server.port 8507 --server.headless true
```

Notas:

- `run_app` entrena automaticamente si no encuentra modelos en `models/`.
- Si faltan modelos y falta `data/raw/train.csv`, el script termina con un mensaje claro.
- usar `python -m streamlit` dentro del entorno virtual evita conflictos de PATH con un `streamlit` global.

## Entrenamiento

```bash
python -m src.train
```

Alternativa recomendada para evitar conflictos de entorno:

```bash
.venv/Scripts/python.exe -m src.train
```

Salida esperada:

- Modelos en `models/linear_regression.joblib` y `models/random_forest.joblib`
- Metricas en `reports/metrics_comparison.csv`

## Prediccion por consola

```bash
python -m src.predict --tamano 120 --ubicacion NAmes --habitaciones 3
```

Notas:

- Se usan las entradas del usuario exigidas por el enunciado (`tamano`, `ubicacion`, `habitaciones`).
- La inferencia se ejecuta con todas las variables definidas en `data/features.json`.
- El script completa automaticamente el resto de variables con defaults (mediana/moda) tomados de `data/raw/train.csv`.

## Diagnostico de variables

Para generar reportes de correlacion, importancia de Random Forest, frecuencias categoricas y columnas con baja informacion:

```bash
python scripts/feature_diagnostics.py
```

Salida esperada en `reports/diagnostics/`:

- `estadisticas_numericas.csv`
- `estadisticas_categoricas.csv`
- `correlacion_saleprice.csv`
- `importancia_rf.csv`
- `frecuencias_categoricas.csv`
- `baja_varianza.csv`
- `recomendaciones_features.md`

## Demo interactiva (sistema obligatorio)

```bash
python -m streamlit run app/streamlit_app.py
```

## Flujo del proyecto

1. Ingesta de datos desde `data/raw/train.csv`.
2. Limpieza y transformacion (`src/preprocessing.py`).
3. Entrenamiento de Regresion Lineal y Random Forest (`src/train.py`).
4. Evaluacion comparativa con MAE, MSE, RMSE y R2.
5. Prediccion en CLI o Streamlit (`src/predict.py`, `app/streamlit_app.py`).
