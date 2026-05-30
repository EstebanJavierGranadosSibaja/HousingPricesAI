# Documento Técnico — Predicción de Precio de Viviendas

**Universidad Nacional de Costa Rica — Sede Regional Brunca, Campus Pérez Zeledón**
**Curso:** Inteligencia Artificial · **Proyecto Final**
**Estudiante:** Esteban Granados Sibaja
**Tipo de problema:** Regresión (aprendizaje supervisado)
**Dataset:** Ames Housing (Kaggle — *House Prices: Advanced Regression Techniques*)

---

## 1. Formulación del problema

### 1.1 Definición

El sistema estima el **precio de venta de una vivienda** (`SalePrice`) a partir de
sus características físicas, de calidad y de ubicación. Como el objetivo es una
cantidad monetaria **continua**, el problema es de **regresión** y no de
clasificación.

### 1.2 Variables

- **Variable objetivo (y):** `SalePrice` — precio final de venta (continuo).
- **Variables predictoras (X):** 22 variables seleccionadas:
  - **16 numéricas:** `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`,
    `LotArea`, `YearBuilt`, `YearRemodAdd`, `OverallQual`, `OverallCond`,
    `GarageCars`, `GarageArea`, `FullBath`, `HalfBath`, `BedroomAbvGr`,
    `TotRmsAbvGrd`, `Fireplaces`.
  - **6 categóricas:** `MSZoning`, `Neighborhood`, `KitchenQual`, `BsmtQual`,
    `ExterQual`, `Foundation`.

### 1.3 Justificación técnica del enfoque

- El objetivo es numérico y continuo → métricas de error continuo (MAE, MSE,
  RMSE, R²).
- Existe una relación funcional entre los atributos de la vivienda y su precio.
- Se comparan **dos modelos complementarios**: Regresión Lineal (línea base
  interpretable) y Random Forest (no lineal, capta interacciones).
- **Meta operativa:** minimizar el RMSE en datos no vistos manteniendo un R²
  alto y estable bajo validación cruzada.

### 1.4 Encuadre como agente inteligente (PEAS)

| Componente | En este proyecto |
|---|---|
| **Performance** | RMSE/MAE bajos y R² alto en datos no vistos |
| **Environment** | Mercado inmobiliario de Ames, Iowa (2006–2010) |
| **Actuators** | Precio estimado + comparación entre modelos |
| **Sensors** | Vector de 22 características ingresadas por el usuario |

El entorno es **parcialmente observable** (no se conocen todos los factores del
mercado), **estocástico**, **estático** respecto a una consulta individual y
**continuo** en sus variables y salida.

---

## 2. Dataset y análisis exploratorio (EDA)

El dataset original tiene **1460 filas y 81 columnas**. Se redujo a 22 variables
con criterio técnico (señal predictiva, baja redundancia, interpretabilidad).
Hallazgos principales del EDA (detalle y gráficos en el notebook):

| Hallazgo | Evidencia | Decisión derivada |
|---|---|---|
| `SalePrice` con fuerte asimetría positiva | skew = **1.88** (original) → **0.12** con `log1p`; media 180.921 > mediana 163.000 | Tratar outliers; considerar transformación log como mejora futura |
| Predictores más fuertes | Mayor correlación: `OverallQual`, `GrLivArea`, `GarageCars/Area`, `TotalBsmtSF` | Confirma la selección de variables |
| Multicolinealidad | `GarageCars`–`GarageArea`, `TotRmsAbvGrd`–`GrLivArea` | Escalado para la Regresión Lineal; descartar redundantes |
| Outliers | **31** viviendas sobre el límite IQR de `GrLivArea`; casas grandes con precio bajo | Winsorización (clip 1%/99%) en lugar de eliminar filas |
| Sesgo geográfico | El vecindario más caro vale **3.58×** el más barato | `Neighborhood` es predictor clave y fuente de sesgo |
| No linealidad | Saltos de precio crecientes según `OverallQual` | Justifica usar Random Forest |

### 2.1 Tratamiento de outliers (justificación)

Se decidió **winsorizar** (recortar al percentil 1/99 con `QuantileClipper`) en
lugar de eliminar registros porque: (1) la muestra es pequeña (1460 filas) y
eliminar descarta información real; (2) el clipping acota el impacto de los
extremos sin perder la observación; (3) los límites se ajustan **solo con el
conjunto de entrenamiento** dentro del pipeline, evitando fuga de información.

---

## 3. Preprocesamiento

Todo el preprocesamiento vive dentro de un **`Pipeline` de scikit-learn con
`ColumnTransformer`**, lo que garantiza reproducibilidad y reutilización idéntica
en entrenamiento, validación cruzada, inferencia CLI y aplicación Streamlit (sin
fuga de información).

Flujo: `DomainImputer → RareCategoryGrouper → QuantileClipper → ColumnTransformer`.

| Etapa | Qué hace | Justificación |
|---|---|---|
| `DomainImputer` | Reglas de dominio (sin sótano/garaje ⇒ valores 0 y etiquetas explícitas) + flags `has_garage`/`has_bsmt` | Los nulos son **estructurales**, no aleatorios |
| Imputación numérica | `SimpleImputer(median)` | Robusta frente a asimetría |
| Imputación categórica | `SimpleImputer(constant="Missing")` | Conserva la señal de "ausencia" |
| `RareCategoryGrouper` | Agrupa categorías con frecuencia < 1% en "Other" | Evita columnas dummy inestables |
| `QuantileClipper` | Winsorización 1%/99% | Control de outliers (ver 2.1) |
| `OneHotEncoder(handle_unknown="ignore")` | Codifica categóricas | Tolera categorías no vistas en inferencia |
| `StandardScaler` | Escala numéricas (solo en Regresión Lineal) | Necesario para el modelo lineal; innecesario para árboles |

**División de datos:** 80% entrenamiento / 20% prueba (`random_state=42`). La
validación cruzada (KFold=5) se ejecuta **solo sobre el conjunto de entrenamiento**.

---

## 4. Modelos implementados

- **Modelo 1 — Regresión Lineal:** línea base interpretable y de baja varianza.
- **Modelo 2 — Random Forest:** modelo no lineal, capta interacciones sin
  ingeniería manual extensa.

**Ajuste de hiperparámetros:** `GridSearchCV` con KFold=5 y scoring por RMSE.
Grillas: `fit_intercept`/`positive` (lineal); `n_estimators`, `max_depth`,
`min_samples_split`, `min_samples_leaf`, `max_features` (Random Forest).

---

## 5. Evaluación experimental

### 5.1 Métricas en el conjunto de prueba (modelos tras tuning)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** (ganador) | ≈ 17.065 | **28.253** | **0.896** |
| Regresión Lineal | — | 31.947 | 0.866 |

El **Random Forest gana** por menor RMSE en prueba. El MAPE del modelo ganador es
**10.7%**, indicador interpretable del error relativo promedio.

### 5.2 Análisis de residuos y errores

- **Residuos:** media cercana a 0 (predicciones aproximadamente insesgadas), pero
  con **heterocedasticidad** (la dispersión crece en precios altos). El histograma
  de residuos es simétrico con colas pesadas por viviendas atípicas.
- **Reales vs predichos:** buena alineación con la diagonal; tendencia a
  **subestimar viviendas de lujo** (pocos ejemplos).
- **Errores grandes:** se concentran en propiedades caras/atípicas → el sistema es
  más confiable en el rango de precio medio.

### 5.3 Diagnóstico de sobreajuste (RMSE train vs test)

| Modelo | RMSE train | RMSE test | Gap |
|---|---|---|---|
| Regresión Lineal | 28.245 | 31.947 | 3.702 (11.6%) |
| Random Forest | 11.043 | 28.253 | 17.210 (60.9%) |

El Random Forest presenta un gap considerable (memoriza parte del ruido del
entrenamiento), **pero la validación cruzada y la regularización por
hiperparámetros mantienen el desempeño en prueba superior al modelo lineal**. Por
eso la selección final se basa en el RMSE de **test**, no de train.

---

## 6. Sistema interactivo

- **Notebook:** widget `ipywidgets` para simular escenarios y comparar predicciones.
- **Aplicación Streamlit (`app/`):** modo rápido (3 variables + defaults) y modo
  completo (22 variables), con validación de entradas, manejo de errores,
  simulador A/B de mejoras, consenso ponderado entre modelos, contexto por
  percentiles e historial de corridas.

---

## 7. Análisis crítico

1. **Limitaciones del dataset:** una sola ciudad (Ames, Iowa) y un período
   histórico (2006–2010); los patrones no se trasladan automáticamente a otras
   regiones ni a condiciones actuales.
2. **Sobreajuste:** Random Forest tiende a sobreajustar con muestras pequeñas; se
   monitorea con gap train-test y validación cruzada.
3. **Sesgos:** geográfico (concentración en ciertos vecindarios) y temporal.
4. **Escalabilidad:** ampliar variables y grillas crece de forma no lineal en
   costo computacional; convendría búsquedas más eficientes (p. ej. aleatoria).
5. **Consideraciones éticas:** la predicción no es verdad absoluta de mercado;
   usar `Neighborhood` de forma determinista puede reforzar **desigualdades
   territoriales**. El sistema debe presentarse como apoyo a la decisión, no como
   valoración oficial, y con cautela explícita en propiedades de lujo o atípicas.

---

## 8. Conclusiones

- Se construyó una solución de regresión **completa, reproducible y modular**:
  formulación → EDA → preprocesamiento con Pipeline → 2 modelos → validación
  cruzada → tuning → evaluación con análisis de residuos → sistema interactivo →
  análisis crítico.
- El **Random Forest** es el modelo final (RMSE ≈ 28.253, R² ≈ 0.896, MAPE 10.7%),
  superando a la Regresión Lineal por su capacidad de modelar no linealidades.
- La Regresión Lineal aporta **interpretabilidad y estabilidad**; ambos se
  muestran al usuario para una decisión informada.
- **Mejoras futuras:** transformar el objetivo con `log1p` para mitigar la
  heterocedasticidad, incorporar más variables del dataset y validar el modelo
  con datos de otras regiones/períodos.

---

## 9. Referencias

- Dataset: <https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>
- Métricas de regresión (MAE, MSE, RMSE): <https://www.datacamp.com/es/tutorial/rmse>
