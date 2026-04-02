# Justificación Paso a Paso: Predicción de Precios de Viviendas (Random Forest)

Este documento detalla y justifica cada una de las fases implementadas en el script principal (`test.py`) para resolver el problema de predicción de precios.

## Fase 1: Librerías y Configuración

**Qué se hizo:** Se importaron bibliotecas estándar para Ciencia de Datos: `pandas` para manipulación de DataFrames, `numpy` para cálculos y transformaciones, y modulos de `scikit-learn` para modelado (Random Forest), métricas (RMSE) y partición de datos.
**Justificación:** Son las herramientas óptimas en Python. `RandomForestRegressor` es ideal porque maneja bien relaciones no lineales y es robusto ante valores atípicos (outliers) sin necesidad de escalar los datos rígidamente.

## Fase 2: Carga de Datos y Selección de Variables

**Qué se hizo:** Se cargaron los archivos `train.csv` y `test.csv`. Se filtró el dataset original de 81 columnas a solo 20 variables críticas (12 numéricas y 8 categóricas).
**Justificación:** Según los análisis previos (`importancia_rf.csv` y `correlacion_SalePrice.csv`), usar todas las variables añade "ruido". Seleccionamos únicamente variables con alta correlación con el precio o alta varianza demostrada (ej. `OverallQual`, `GrLivArea`, `Neighborhood`), descartando aquellas con un alto porcentaje de valores nulos o que aportaban nula información (ej. `Alley`, `PoolQC`). Esto agiliza el aprendizaje y previene el sobreajuste (overfitting).

## Fase 3: Preprocesamiento de Datos

**Qué se hizo:**

1. Se unieron temporalmente los datos de Train y Test.
2. Se imputaron los nulos usando la **mediana** para numéricos y la **moda** para categóricos.
3. Se aplicó *One-Hot Encoding* (`pd.get_dummies`) a las categóricas.
**Justificación:**

- Al unir Test y Train durante el preprocesamiento, garantizamos que el *One-Hot Encoding* genere exactamente las mismas columnas en ambos sets (evitando errores si una categoría existe en Test pero no en Train).
- La **mediana** se usa en vez de la media porque no se ve afectada por valores extremos (casas anormalmente grandes).
- Las categorías requieren ser números (0 o 1) para que el Random Forest pueda matemáticamente procesarlas.

## Fase 4: Transformación del Target (SalePrice)

**Qué se hizo:** Se aplicó una transformación logarítmica (`np.log1p`) a la variable objetivo `SalePrice`.
**Justificación:** Los precios de las viviendas suelen tener un "sesgo a la derecha" (unas pocas casas son extremadamente caras, estirando la gráfica). Esta transformación convierte la distribución de precios en una forma más normal (campana de Gauss), lo que mejora drásticamente la capacidad de modelo para aprender los patrones sin obsesionarse con los precios extremos.

## Fase 5: División de Entrenamiento y Validación

**Qué se hizo:** Se segmentó el conjunto de entrenamiento (train) en 80% para entrenar y 20% para validar (`train_test_split`).
**Justificación:** Si evaluamos el modelo con los mismos datos con los que aprendió, nos engañaría dándonos una nota perfecta pero fallaría en la vida real. Separar un 20% "ciego" ayuda a medir el RMSE real (Error Cuadrático Medio) que tendría con clientes nuevos.

## Fase 6: Entrenamiento del Modelo

**Qué se hizo:** Se creó el modelo `RandomForestRegressor(n_estimators=200, max_depth=9)` y se ajustó (`.fit()`).
**Justificación:**

- `n_estimators=200`: Un "bosque" de 200 árboles de decisión es un equilibrio excelente entre precisión y tiempo de procesamiento.
- `max_depth=9`: Limitar la profundidad máxima del árbol evita que el modelo memorice el ruido de los casos específicos (overfitting) y le obliga a aprender reglas generales.

## Fase 7: Evaluación del Modelo (RMSE)

**Qué se hizo:** Se predijeron los precios del set de validación, se invirtió el logaritmo con la exponencial (`np.expm1`) y se calculó el RMSE.
**Justificación:** Como habíamos transformado los precios a su logaritmo, las predicciones salen en formato logarítmico. Invertirlas (`expm1`) nos permite entender el error en **dólares reales ($)**. El RMSE nos dice, en promedio, qué tanto se equivoca el modelo en el precio de una casa.

## Fase 8: Predicción Final y Guardado

**Qué se hizo:** Se generó la predicción usando el set de `test.csv` limpio, se invirtió su logaritmo, y se creó un CSV final (`submission_rf.csv`) con las columnas `Id` y `SalePrice`.
**Justificación:** Prepara la salida al formato exacto requerido por competencias analíticas (Kaggle) o para ser consumido por un sistema / base de datos final del cliente.
