# Proyecto Final – Implementación de IA

## 1. Descripción General

El proyecto consiste en diseñar, implementar y evaluar una solución funcional basada en Inteligencia Artificial que modele un problema real.

Debe incluir:
- Aplicación de técnicas de Machine Learning, PLN o Visión por Computador.
- Implementación de al menos **dos modelos comparables**.
- Evaluación experimental del desempeño.
- Interacción directa con el usuario.

---

## 2. Objetivo General

Desarrollar un sistema de IA funcional que resuelva un problema real mediante:
- Modelado completo
- Evaluación experimental
- Análisis crítico

---

## 3. Propuesta de Proyecto: Predicción de Precio de Viviendas

### Tipo
Regresión

### Descripción
Predecir el precio de una vivienda en función de sus características principales.

### Variables de entrada
- Tamaño (m²)
- Ubicación
- Número de habitaciones

### Variable de salida
- Precio estimado de la vivienda

### Modelos a implementar
- Regresión Lineal
- Random Forest

### Resultado esperado
El sistema permitirá que el usuario ingrese:
- Tamaño
- Ubicación
- Habitaciones

Y devolverá:
- Precio estimado
- Comparación entre modelos (rendimiento y predicción)

---

## 4. Componentes del Proyecto

### 4.1 Formulación del Problema
- Definición clara del problema
- Tipo de aprendizaje: Supervisado (Regresión)
- Identificación de variables
- Justificación técnica

---

### 4.2 Dataset
- Dataset real (ej: Kaggle, Zillow, etc.)
- Descripción de variables
- Análisis exploratorio (EDA)
- Identificación de limitaciones

---

### 4.3 Preprocesamiento
- Limpieza de datos
- Transformación de variables
- Normalización/Escalado
- División:
  - Train
  - Test

---

### 4.4 Implementación de Modelos
- Modelo 1: Regresión Lineal
- Modelo 2: Random Forest
- Ajuste básico de hiperparámetros
- Justificación de elección

---

### 4.5 Evaluación Experimental

Métricas de regresión:
- MAE (Error absoluto medio)
- MSE (Error cuadrático medio)
- RMSE
- R²

Incluye:
- Comparación entre modelos
- Interpretación de resultados

---

### 4.6 Sistema Interactivo

Debe permitir:
- Ingreso manual de datos
- Predicción en tiempo real

Opciones:
- Notebook (input interactivo)
- Interfaz con Streamlit o Gradio

---

### 4.7 Análisis Crítico

- Limitaciones del modelo
- Riesgo de overfitting
- Sesgos en los datos
- Escalabilidad
- Consideraciones éticas

---

## 5. Entregables

- Notebook (.ipynb)
- Documento técnico (PDF)
- Presentación
- Dataset
- Sistema interactivo funcional
- Defensa oral

---

## 6. Estructura del Notebook

- Introducción
- Problema
- Dataset
- EDA
- Preprocesamiento
- Modelo 1
- Modelo 2
- Evaluación
- Sistema interactivo
- Análisis crítico
- Conclusiones

---

## 7. Evaluación (Resumen)

| Criterio | Peso |
|--------|------|
| Formulación del problema | 10% |
| Dataset y preprocesamiento | 10% |
| Implementación técnica | 20% |
| Comparación de modelos | 15% |
| Evaluación y métricas | 15% |
| Análisis crítico | 10% |
| Documentación | 10% |
| Defensa oral | 10% |

---

## 8. Idea Clave del Proyecto

Dos modelos compiten.  
Uno simple (Lineal), otro potente (Random Forest).  
El usuario decide… pero los datos hablan.
