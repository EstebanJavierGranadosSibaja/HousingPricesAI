# ProyectoAI - Prediccion de Precio de Viviendas

Proyecto final de IA (regresion) para predecir precios de viviendas con dos modelos comparables:

- Regresion Lineal
- Random Forest

Incluye notebook principal, entrenamiento reproducible, evaluacion con metricas (MAE, MSE, RMSE, R2) y sistema interactivo con Streamlit.

## Modelo final seleccionado

**Random Forest fue seleccionado como modelo final del sistema porque obtuvo el mejor
desempeno predictivo en el conjunto de prueba.** La Regresion Lineal con transformacion
logaritmica (`log1p`) se reconoce como una alternativa metodologicamente solida, mas
estable y con menor riesgo de sobreajuste, pero no fue seleccionada como modelo operativo
debido a que presento un desempeno predictivo inferior en test. La diferencia de
desempeno entre modelos no es estadisticamente significativa (prueba *t* pareada,
p ≈ 0.13); por eso se opta por el mejor estimador puntual en test (Random Forest) como
modelo desplegado, mostrando ambos modelos al usuario para una decision informada.

## Dependencias

- Probado con **Python 3.11**.
- `requirements.txt`: dependencias minimas para ejecutar entrenamiento, prediccion y app (con versiones fijadas para reproducibilidad).
- `requirements-notebook.txt`: dependencias adicionales para el notebook (matplotlib, seaborn, jupyter, ipywidgets).

## Estructura

```text
HousingPricesAI/
├── data/
│   ├── features.json        # manifest de 22 variables (fuente de verdad)
│   └── raw/                 # train.csv, test.csv, sample_submission.csv
├── notebooks/
│   └── 01_housing_prices_proyecto_final.ipynb
├── src/
│   ├── preprocessing.py     # limpieza, transformacion, pipeline
│   ├── train.py             # entrenamiento y evaluacion de ambos modelos
│   └── predict.py           # prediccion por CLI
├── app/
│   ├── streamlit_app.py     # interfaz interactiva (4 tabs)
│   ├── analytics.py         # metricas, importancias, percentiles
│   ├── catalog.py           # etiquetas amigables de features
│   ├── config.py            # constantes y metadatos de features
│   ├── services.py          # carga de modelos, presets de entrada
│   ├── ui.py                # componentes visuales y CSS
│   ├── validation.py        # sanitizacion de entradas
│   └── explainability/      # capa de explicabilidad con SHAP + LLM
│       ├── shap_explainer.py
│       ├── llm_provider.py  # OllamaProvider, MockProvider
│       └── explanation_service.py
├── models/                  # modelos entrenados (.joblib) — gitignoreados
├── reports/
│   ├── metrics_comparison.csv
│   └── ablation_study.csv
├── docs/
│   ├── EnunciadoProyecto.md
│   ├── document.md
│   ├── DEFENSA/             # documentos de entrega academica
│   └── dataset/             # descripcion de variables (EN + ES)
├── tests/
│   ├── test_pipeline.py     # 8 tests: pipeline, transformadores, metricas
│   └── test_explainability.py # 29 tests: SHAP, LLM providers, service
├── scripts/
│   ├── feature_diagnostics.py
│   ├── ablation_study.py
│   ├── init_ollama.ps1      # setup Ollama en Windows
│   ├── init_ollama.sh       # setup Ollama en Linux/macOS
│   ├── setup_env.ps1
│   ├── run_app.ps1
│   ├── setup_env.sh
│   └── run_app.sh
├── .env.example             # plantilla de configuracion LLM
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
- Metricas en `reports/metrics_comparison.csv` (fuente unica de verdad de las metricas)

Por defecto entrena la configuracion base. Para reproducir el ajuste de
hiperparametros del notebook (GridSearchCV con KFold=5) y persistir el modelo
ajustado:

```bash
python -m src.train --tune
```

## Prediccion por consola

```bash
python -m src.predict --tamano 120 --ubicacion NAmes --habitaciones 3
```

Notas:

- Se usan las entradas del usuario exigidas por el enunciado (`tamano`, `ubicacion`, `habitaciones`).
- La inferencia se ejecuta con las 22 variables definidas en `data/features.json`.
- El script completa automaticamente el resto de variables con defaults (mediana/moda) tomados de `data/raw/train.csv`.

## Pruebas

Suite completa: 37 tests en total (8 de pipeline ML + 29 de capa de explicabilidad).

```bash
python -m pytest tests/ -v
```

O usando unittest directamente:

```bash
python -m unittest discover -s tests -v
```

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

## Explicacion IA (SHAP + Ollama)

La app incluye una pestaña de explicabilidad que usa SHAP TreeExplainer para identificar
las variables con mayor impacto en cada prediccion, y un LLM local (Ollama) para
generar una explicacion en lenguaje natural en espanol.

Setup de Ollama (primera vez):

Windows:

```powershell
.\scripts\init_ollama.ps1
```

Linux/macOS:

```bash
bash scripts/init_ollama.sh
```

Configura el modelo en `.env` (copia `.env.example`):

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:8b       # o llama3.2:3b para equipos con menos RAM
OLLAMA_TIMEOUT=30
```

Si Ollama no esta disponible, la app cae automaticamente a un modo plantilla (MockProvider)
que genera una explicacion basada en los factores SHAP sin requerir LLM.

## Demo interactiva (sistema obligatorio)

```bash
python -m streamlit run app/streamlit_app.py
```

La app tiene 4 pestanas: comparacion de modelos, variables mas influyentes, rendimiento
(MAE, MSE, RMSE, R2) y explicacion IA (SHAP + lenguaje natural). El panel lateral ajusta
las caracteristicas clave; el resto se completa con valores tipicos automaticamente.

## Flujo del proyecto

1. Formulacion del problema (variable objetivo, predictores, tipo de tarea).
2. Ingesta de datos desde `data/raw/train.csv`.
3. Analisis exploratorio (EDA): distribucion del precio, correlaciones, outliers y sesgo geografico, cada uno con interpretacion tecnica.
4. Limpieza y transformacion con Pipeline + ColumnTransformer (`src/preprocessing.py`).
5. Entrenamiento de Regresion Lineal y Random Forest (`src/train.py`).
6. Validacion cruzada (KFold=5) y ajuste de hiperparametros (GridSearchCV) en el notebook.
7. Evaluacion comparativa con MAE, MSE, RMSE y R2, mas analisis de residuos, errores grandes, importancia de variables y diagnostico de sobreajuste.
8. Prediccion en CLI o Streamlit (`src/predict.py`, `app/streamlit_app.py`).

El detalle metodologico completo esta en `notebooks/01_housing_prices_proyecto_final.ipynb` y el documento tecnico en `docs/document.md`.
