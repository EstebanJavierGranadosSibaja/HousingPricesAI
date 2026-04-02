# Guía del proceso: House Prices Advanced Regression Techniques

Esta guía describe el flujo de trabajo realizado en el notebook "house-prices-advanced-regression-techniques.ipynb" para abordar el problema de predicción de precios de viviendas usando técnicas avanzadas de regresión.

## 1. Planteamiento del problema

- El objetivo es predecir el precio de venta de viviendas a partir de un conjunto de variables predictoras (numéricas y categóricas) del dataset de Kaggle "House Prices: Advanced Regression Techniques".

## 2. Importación de librerías

- Se utilizan librerías como pandas, numpy, matplotlib, seaborn y scikit-learn para manipulación de datos, visualización, preprocesamiento y modelado.

## 3. Carga y exploración de datos

- Se cargan los archivos de datos de entrenamiento y prueba.
- Se exploran las primeras filas, dimensiones, nombres de columnas y tipos de datos.
- Se identifican variables relevantes y se analizan valores nulos.

## 4. Análisis exploratorio

- Se visualizan distribuciones de variables y relaciones con el precio de venta.
- Se exploran correlaciones y se identifican variables importantes.

## 5. Preprocesamiento de datos

- Se realiza limpieza de datos: manejo de valores nulos, codificación de variables categóricas, escalado de variables numéricas, generación de nuevas variables si es necesario.
- Se separan variables predictoras (X) y variable objetivo (y).

## 6. División de datos

- Se dividen los datos en conjuntos de entrenamiento y validación para evaluar el desempeño del modelo.

## 7. Entrenamiento de modelos

- Se entrenan modelos de regresión como Linear Regression, Ridge, Lasso, Random Forest, XGBoost, entre otros.
- Se ajustan hiperparámetros y se evalúan diferentes enfoques.

## 8. Evaluación y comparación de modelos

- Se evalúan los modelos usando métricas como RMSE, MAE y R².
- Se comparan los resultados y se selecciona el mejor modelo según el desempeño en validación.

## 9. Predicción y generación de archivo de envío

- Se utiliza el modelo seleccionado para predecir los precios en el conjunto de prueba.
- Se genera el archivo de envío para Kaggle con los resultados.

## 10. Conclusiones

- Se discuten los resultados obtenidos, las variables más relevantes y posibles mejoras futuras.

---

Esta guía sirve como referencia rápida para entender el flujo de trabajo típico en un proyecto de Machine Learning aplicado a regresión avanzada de precios de viviendas.
