Actúa como un arquitecto senior en Machine Learning especializado en proyectos académicos que también funcionan como portafolio profesional.

Tu objetivo es diseñar una estructura de proyecto que maximice la calificación académica, pero que también sea lo suficientemente limpia, clara y profesional como para incluirse en un portafolio.

IMPORTANTE:

* Prioriza cumplir el enunciado al 100%
* NO agregues complejidad innecesaria
* Mantén una estructura limpia pero con criterio profesional
* Piensa como profesor (evaluación) y como reclutador (portafolio)

---

## ENUNCIADO DEL PROYECTO

"El enuncido está en el archivo EnunciadoProyecto.md"

---

## CONTEXTO

Proyecto de regresión para predicción de precios de viviendas con:

* 2 modelos obligatorios:

  * Regresión Lineal
  * Random Forest
* Métricas:

  * MAE
  * MSE
  * RMSE
  * R²
* Sistema interactivo para el usuario
* Notebook como pieza principal de entrega

---

## OBJETIVO

Diseñar una estructura que:

1. Maximice la nota (alineación perfecta con rúbrica)
2. Sea clara y fácil de evaluar
3. Se vea profesional en GitHub (portafolio)
4. Mantenga simplicidad (sin sobreingeniería)

---

## TAREAS

### 1. Estructura del proyecto (CLAVE)

Define una estructura en formato árbol que sea:

* Clara
* Minimalista
* Profesional

Debe incluir SOLO lo necesario, por ejemplo:

* notebooks/
* data/
* src/ (ligero, no sobrecargado)
* models/
* reports/
* app/ o demo/

Cada carpeta debe tener:

* Propósito claro
* Uso real dentro del proyecto

---

### 2. Estructura del notebook (CRÍTICO PARA LA NOTA)

Define una estructura perfecta del notebook alineada EXACTAMENTE con la rúbrica:

* Introducción
* Formulación del problema
* Dataset
* EDA
* Preprocesamiento
* Modelo 1 (Lineal)
* Modelo 2 (Random Forest)
* Evaluación comparativa
* Sistema interactivo
* Análisis crítico
* Conclusiones

Para cada sección indica:

* Qué incluir
* Qué mostrar (gráficos, tablas, código)
* Qué evitar

---

### 3. Flujo del sistema (END-TO-END)

Describe el flujo ideal:

* Ingesta de datos
* Limpieza
* Transformación
* Entrenamiento (ambos modelos)
* Evaluación
* Comparación
* Predicción

Debe ser simple, lineal y entendible.

---

### 4. Código mínimo profesional (SIN EXCESO)

Define qué archivos sí valen la pena en `src/`:

Ejemplo esperado:

* preprocessing.py
* train.py
* predict.py

Evita:

* arquitectura compleja
* patrones innecesarios

---

### 5. Sistema interactivo (OBLIGATORIO)

Propón la forma MÁS simple y efectiva:

* Notebook interactivo o Streamlit
* Inputs:

  * tamaño
  * ubicación
  * habitaciones
* Outputs:

  * predicción por modelo
  * comparación

Debe ser fácil de demostrar en evaluación.

---

### 6. Evaluación experimental

Define cómo presentar:

* Métricas claras en tabla
* Comparación directa entre modelos
* Visualizaciones simples pero efectivas

---

### 7. Análisis crítico (DONDE SE GANA LA NOTA)

Explica qué incluir:

* Overfitting vs underfitting
* Limitaciones del dataset
* Sesgos
* Qué mejorarías con más tiempo

---

### 8. Toques de portafolio (SIN ROMPER EL ALCANCE)

Sugiere mejoras ligeras que suman valor visual/profesional:

* README bien estructurado
* Capturas del sistema interactivo
* Explicación clara del problema

SIN agregar:

* despliegues complejos
* microservicios
* pipelines industriales

---

## FORMATO DE RESPUESTA

Entrega:

1. Estructura del proyecto (árbol)
2. Explicación breve de cada carpeta
3. Estructura ideal del notebook
4. Flujo del sistema
5. Recomendaciones prácticas

Sé directo, práctico y enfocado en maximizar impacto académico + valor de portafolio.
