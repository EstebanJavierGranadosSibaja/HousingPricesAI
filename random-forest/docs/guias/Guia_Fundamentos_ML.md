# Guía del proceso: Fundamentos de Machine Learning

Esta guía describe paso a paso el proceso realizado en el notebook "Fundamentos de Machine Learning" para construir un modelo de regresión, desde la carga de datos hasta la evaluación y comparación de modelos.

## 1. Planteamiento del problema

- Se parte de un problema real: predecir el consumo energético de edificios (Cooling Load) a partir de sus características físicas.
- Se define la variable objetivo y las variables predictoras.

## 2. Importación de librerías

- Se utilizan librerías como pandas, numpy, matplotlib, seaborn y scikit-learn para manipulación de datos, visualización y modelado.

## 3. Carga y exploración del dataset

- Se carga el dataset desde un archivo Excel.
- Se exploran las primeras filas, la dimensión, los nombres de columnas y los tipos de datos.
- Se identifican variables numéricas y categóricas.

## 4. Análisis exploratorio

- Se visualizan relaciones entre variables mediante gráficos de dispersión y mapas de correlación.
- Se analiza la correlación entre variables predictoras y la variable objetivo.

## 5. Preparación de datos

- Se definen las variables predictoras (X) y la variable objetivo (y).
- Se dividen los datos en conjuntos de entrenamiento, validación y prueba.

## 6. Entrenamiento del modelo

- Se entrena un modelo base de regresión lineal.
- Se explica el concepto de función de pérdida (MSE) y minimización empírica del riesgo.

## 7. Evaluación del modelo

- Se evalúa el modelo usando métricas como MSE, MAE y R² en los conjuntos de validación y prueba.
- Se visualizan los resultados y se interpretan los coeficientes del modelo.

## 8. Regularización y comparación de modelos

- Se entrenan modelos con regularización L2 (Ridge) y L1 (Lasso).
- Se comparan los errores de los diferentes modelos para analizar el impacto de la regularización.

## 9. Conclusiones

- Se discuten los resultados, el concepto de underfitting y overfitting, y la importancia de la regularización.
- Se resalta la importancia de comparar diferentes enfoques y métricas para seleccionar el mejor modelo.

---

Esta guía sirve como referencia rápida para entender el flujo de trabajo típico en un proyecto de Machine Learning supervisado de regresión.
