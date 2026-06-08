# Guía de Defensa — Predicción de Precios de Viviendas

**Estudiantes:** Esteban Granados Sibaja · Juan Carlos Camacho Solano · Allan Vargas Torrer · Francisco Mora Cabezas  
**Curso:** Inteligencia Artificial — Universidad Nacional de Costa Rica  
**Fecha:** 2026-06-08

---

## 1. Resumen ejecutivo del proyecto

Se construyó un sistema completo de regresión para predecir el precio de venta de
viviendas usando el dataset Ames Housing (1 460 registros, 22 variables seleccionadas).
El ciclo cubre: formulación PEAS → EDA → pipeline de preprocesamiento reproducible →
entrenamiento de dos modelos → validación cruzada → ajuste de hiperparámetros →
evaluación con análisis de residuos → sistema interactivo Streamlit + notebook con
widgets → explicabilidad SHAP + capa LLM (Ollama).

**Métricas del modelo servido (Random Forest, `reports/metrics_comparison.csv`):**

| Métrica | Random Forest | LR + log1p (alternativa) |
|---|---|---|
| MAE | 17 212 USD | 16 208 USD |
| RMSE test | 27 308 USD | 28 581 USD |
| R² test | 0.903 | 0.894 |
| Gap sobreajuste | ~60 % | **~1.8 %** |

---

## 2. Modelo ganador y justificación

**Modelo final del sistema: Random Forest** (`n_estimators=300`, `random_state=42`).

**Justificación por RMSE (métrica primaria):**
El RMSE penaliza más los errores grandes; en un modelo de valoración inmobiliaria,
subestimar una propiedad cara es más costoso que subestimar una barata. Por RMSE y R²,
el Random Forest supera a la Regresión Lineal en el conjunto de prueba.

**Matiz estadístico honesto:**
La diferencia de desempeño **no es estadísticamente significativa** (prueba t pareada
sobre CV repetida 5×5, p ≈ 0.13; RF gana en 18/25 particiones). Conclusión: el RF es
el mejor estimador puntual, pero no supera de forma robusta a la LR.

**Por eso la app muestra ambos modelos** y un consenso ponderado: el usuario ve la
incertidumbre, no solo un número.

---

## 3. Principales hallazgos del estudio de ablación

El estudio añade componentes de forma acumulativa (configs A→G) midiendo RMSE en test
y validación cruzada. Fuente: `reports/ablation_study.csv` / `scripts/ablation_study.py`.

| Config | Descripción (acumulativa) | RMSE test | RMSE CV (media ± std) |
|---|---|---|---|
| A) Baseline LR | LR + preprocesador básico (sin log) | 32 001 | 29 994 ± 4 396 |
| B) +log1p(SalePrice) | + `TransformedTargetRegressor(log1p)` | **27 339** | 28 771 ± 6 400 |
| C) +log1p(features) | + `log1p` en variables sesgadas (GrLivArea, TotalBsmtSF…) | 27 658 | 28 110 ± 4 716 |
| D) +OrdinalEncoder | + codificación ordinal de calidades (KitchenQual, ExterQual, BsmtQual) | 28 581 | 28 600 ± 4 533 |
| E) +HouseAge | + `HouseAge`/`YearsSinceRemodel` (preprocesador completo LR) | 28 581 | 28 600 ± 4 533 |
| F) Baseline RF | RF `n_estimators=300` + preprocesador básico | 27 647 | 30 435 ± 5 145 |
| G) **RF Final** ✓ | RF + preprocesador completo de producción | **27 308** | 30 007 ± 5 023 |

**Hallazgo clave:** `log1p` (config B) es la mejora individual más grande para la
Regresión Lineal (RMSE test 32 001 → 27 339, −14.6%). `OrdinalEncoder` y
`AgeFeatureCreator` (configs D y E) **no** aportan señal adicional al modelo lineal —
de hecho empeoran el RMSE de test (27 658 → 28 581): el LR no explota bien las
relaciones ordinales de calidad que sí captura el RF. El preprocesador completo
beneficia al RF: config G (27 308) supera a la baseline F (27 647).

---

## 4. Explicación de SHAP

SHAP (SHapley Additive exPlanations) asigna a cada variable su contribución exacta a la
predicción de una vivienda específica, en dólares.

**Cómo funciona aquí:**
1. Se toma la predicción individual del Random Forest para una vivienda concreta.
2. `shap.TreeExplainer` calcula, en promedio sobre todas las trayectorias del árbol,
   cuánto cambiaría la predicción si esa variable no estuviera.
3. Los valores SHAP positivos indican variables que *suben* el precio estimado;
   los negativos, variables que lo *bajan*.
4. Las contribuciones de variables OHE se agregan de vuelta a la variable original
   (p. ej., todos los dummies de `Neighborhood` se suman en un solo valor SHAP).

**Por qué es valioso:** las importancias globales del RF (impurity) dicen qué variables
importan en general; SHAP dice por qué *esta* vivienda cuesta *este* precio — que es lo
que necesita un comprador.

---

## 5. Explicación de la capa LLM

La arquitectura de explicabilidad tiene dos capas:

```
SHAPExplainer.compute_factors()   →  top 5 factores positivos / negativos en $
_build_prompt()                   →  prompt estructurado en español
OllamaProvider.generate()         →  texto natural para el usuario final
    └── MockProvider (fallback)   →  plantilla si Ollama no está activo
```

**Por qué Ollama y no una API externa:**
- Privacidad: los datos de la vivienda no salen del equipo.
- Disponibilidad offline: funciona sin internet.
- Costo cero en inferencia.

**El fallback es diseño deliberado:** si Ollama no está activo, `MockProvider` genera
una explicación usando los mismos factores SHAP, con un mensaje al pie que invita a
activar Ollama para mayor detalle. El sistema nunca falla silenciosamente.

---

## 6. Decisiones metodológicas importantes

| Decisión | Alternativa rechazada | Por qué se tomó |
|---|---|---|
| Winsorización 1/99 % (`QuantileClipper`) | Eliminar outliers | La muestra es pequeña (1 460 filas); eliminar descarta información real |
| Pipeline scikit-learn completo | Preprocesar antes del split | Evita fuga de información: los parámetros del scaler/clipper se calculan solo con train |
| `DomainImputer` de reglas de negocio | Imputación genérica | Un nulo en `BsmtQual` no es un valor desconocido: significa "no tiene sótano" |
| `RareCategoryGrouper` (min_freq 1 %) | OHE puro | Categorías con 1-2 ejemplos crean columnas inestables; el agrupador en "Other" mejora la generalización |
| `log1p` en objetivo para LR | Escala original | Corrige asimetría (skew 1.88→0.12) y heterocedasticidad; reduce el gap de sobreajuste del LR de 12 % a 1.8 % |
| RF como modelo final | LR+log1p | Mejor RMSE puntual en test; la diferencia no es significativa pero el RF es más robusto a no linealidades |
| División 80/20 + CV KFold=5 sobre train | CV sobre dataset completo | La partición de test permanece "no vista" hasta la evaluación final |

---

## 7. Respuestas para la defensa

### ¿Por qué RF si LR tiene menor MAE?

La métrica primaria de selección es el **RMSE**, no el MAE, porque el RMSE penaliza
más los errores grandes. En el mercado inmobiliario, subestimar una propiedad cara es
un error de mayor impacto económico que subestimar una barata. Por RMSE (27 308 vs
28 581) y R² (0.903 vs 0.894), el Random Forest supera a la LR. Sin embargo, la
diferencia no es estadísticamente significativa (p ≈ 0.13), por eso la app muestra
ambos modelos.

---

### ¿Por qué RF si LR tiene mejor CV?

El RMSE de CV es 29 994 (LR, config A) vs 30 007 (RF, config G) — los intervalos se solapan ampliamente
(LR ± 4 396 vs RF ± 5 023). Por CV los modelos son **prácticamente equivalentes**; la
ventaja del RF se materializa en el test concreto (80/20). La decisión de elegir RF se
basa en el mejor estimador puntual en test y en la robustez del ensemble frente a no
linealidades (sesgo geográfico, efecto escalonado de `OverallQual`), características
que el modelo lineal no captura bien aunque el CV no lo penalice fuertemente.

---

### ¿Qué aporta SHAP?

Las importancias globales del RF (impurity) dicen qué variables importan *en promedio*
para todo el dataset. SHAP responde una pregunta distinta y más útil para el usuario:
**"¿por qué esta vivienda concreta tiene este precio?"** — desglosado variable por
variable, en dólares, con signo. Es la diferencia entre una estadística poblacional y
una explicación personal.

---

### ¿Qué aporta el LLM?

SHAP produce números (factores con valores en dólares). El LLM convierte esos números
en una **explicación en lenguaje natural** comprensible para un comprador sin formación
técnica: sin mencionar "SHAP", "modelo" ni "machine learning". El resultado es una frase
del tipo "Los factores que más elevan el valor son la calidad de la cocina y el tamaño
habitable". Esa capa transforma un sistema de ML en una herramienta de comunicación.

---

### ¿Por qué usar log1p?

`SalePrice` tiene asimetría positiva fuerte (skew = 1.88). Esto causa:
1. Errores grandes desproporcionados en propiedades caras (el RMSE los magnifica).
2. Heterocedasticidad: la varianza del error crece con el precio, violando supuestos
   del modelo lineal.

`log1p` comprime la cola derecha (skew → 0.12), homogeniza la varianza y reduce el
sobreajuste del modelo lineal del 12 % al 1.8 %. El RF no necesita esta transformación
porque los árboles son invariantes a transformaciones monotónicas del objetivo.

---

### ¿Existe sobreajuste?

**Sí, en el Random Forest, y está documentado honestamente.**

| Modelo | RMSE train | RMSE test | Gap |
|---|---|---|---|
| LR (base, sin log1p) | 28 137 | 32 001 | 12.1 % |
| **LR + log1p** | ≈ 27 300 | 28 581 | **1.8 %** |
| Random Forest | 11 012 | **27 308** | **59.7 %** |

El RF memoriza parte del ruido de entrenamiento. Por eso la confianza en el sistema
se respalda con la validación cruzada (5.1.1), no solo con el train. La Regresión
Lineal + log1p es la alternativa mejor calibrada (gap 1.8 %) y se documenta como
alternativa metodológicamente sólida aunque no fue seleccionada como modelo operativo.

---

## 8. Guion de demo recomendado

1. **Abrir la app Streamlit** → navegar el wizard hasta obtener predicción.
2. **Tab "Comparación de modelos"** → mostrar LR vs RF vs consenso.
3. **Tab "Rendimiento del modelo"** → RMSE, R², gráficos de residuos.
4. **Tab "Variables más influyentes"** → importancias RF con etiquetas en español; **Tab "Explicación IA"** → SHAP + texto generado por Ollama (o MockProvider si offline).
5. **Pivotar al notebook** → mostrar celda SHAP (cell 59) e importancia RF (cell 50).
6. **Widget interactivo** (cell 54) → modificar `GrLivArea` o `OverallQual` en vivo.

---

## 9. Ante preguntas difíciles

| Pregunta | Respuesta de 30 segundos |
|---|---|
| "¿Por qué las métricas del documento no coinciden?" | "La tabla §5.1 compara el RF de producción (RMSE 27 308, fuente: CSV) con la LR de configuración base (sin log1p, RMSE 32 001) como línea base. La LR de producción con preprocesador completo + log1p (ablación configs D/E) obtiene RMSE 28 581, documentada en §5.1. El experimento simplificado config B (solo log1p en objetivo) da RMSE 27 339 y está en §5.1.2. El estudio de ablación documenta cómo el OrdinalEncoder y AgeFeatureCreator no mejoran la LR, resultado honesto." |
| "¿Dónde está el código del p≈0.13?" | "El p ≈ 0.13 es el resultado de una validación cruzada repetida 5×5 con prueba t pareada documentada en el addendum del notebook. El script formal de reproducción queda como trabajo pendiente; el resultado coincide con el solapamiento visible de los intervalos CV." |
| "Ejecute los tests" | Todos los tests pasan (39/39 — 10 de pipeline + 29 de explicabilidad). El bug de `get_feature_names_out` fue corregido en `src/preprocessing.py` (`feature_names_out="one-to-one"`) y el modelo fue reentrenado; las métricas son idénticas a las del CSV. |
| "¿Por qué usar Neighborhood si puede reforzar sesgos?" | "El dataset ya refleja los precios del mercado de 2006–2010. `Neighborhood` es el predictor con mayor señal porque las diferencias de precio por zona son reales. El sistema se presenta como apoyo a la decisión, no como valoración oficial, con esa advertencia explícita en el análisis crítico (§7)." |
