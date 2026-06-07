# 🔍 AUDITORÍA TÉCNICA COMPLETA
## Proyecto: Predicción de Precio de Viviendas (Ames Housing)

> ⚠️ **ACTUALIZACIÓN POSTERIOR (fotografía histórica).** Este documento es la auditoría
> del estado **previo** a las correcciones. Varios hallazgos ya están **resueltos**
> (modelo vs documento, métricas, `has_masonry`, scaler en RF, pins de dependencias).
> Una **validación científica posterior** mostró que la diferencia entre Random Forest y
> Regresión Lineal **no es estadísticamente significativa** (CV repetida, p≈0.13) y que
> `log1p` mejora la **estabilidad** del modelo lineal (sobreajuste 1.8%). **POSTURA
> OFICIAL DE SELECCIÓN:** Random Forest es el **modelo final del sistema** por su mejor
> desempeño predictivo en test; `Regresión Lineal + log1p` es una **alternativa más
> estable, no seleccionada como modelo operativo**. Donde este informe diga "RF gana",
> léase como "RF es el modelo final, con diferencia no significativa frente a la
> alternativa". **Conclusión vigente:** `VALIDACION_CIENTIFICA_FINAL.md` y
> `ALINEAMIENTO_FINAL_CON_RETROALIMENTACION_PROFESOR.md`; remediaciones en
> `CORRECCIONES_REALIZADAS.md`.

> **Rol del auditor:** Staff ML Engineer · Senior Data Scientist · Senior Software Architect · MLOps Engineer · Profesor de IA · Tribunal exigente.
> **Postura:** adversarial. El objetivo es **encontrar fallos**, no justificar el proyecto.
> **Método:** todos los hallazgos fueron **verificados empíricamente** ejecutando código sobre el repositorio, no inferidos.

---

## Resumen del auditor (TL;DR)

El proyecto está **metodológicamente bien planteado** y, sorprendentemente para un trabajo de curso, **no tiene fugas de información graves** gracias al uso disciplinado del `Pipeline`. Sin embargo, presenta **una desconexión seria entre lo que se documenta y lo que se entrega en producción**, **cero reproducibilidad garantizada** (sin pins de dependencias), **una feature muerta**, **redundancia de variables contradictoria con su propia justificación**, y **ausencia total de pruebas**. Ninguno es catastrófico, pero en conjunto bajan la nota de "excelente" a "muy bueno con reservas".

**Hallazgo #1 (el más importante):** el modelo de producción (`models/random_forest.joblib`) **NO es el modelo tuneado del notebook**; es el modelo base sin ajuste de hiperparámetros (`max_depth=None`), y las métricas publicadas en `reports/metrics_comparison.csv` **no coinciden** con las del documento técnico. Lo que se defiende y lo que se ejecuta son cosas distintas.

---

# Sección 1 — Errores Críticos

> Verificados ejecutando el código. Clasificación: 🔴 Crítico · 🟠 Alto · 🟡 Medio · 🟢 Bajo.

### 1.1 🟠 ALTO — Cero reproducibilidad: dependencias sin versión
- **Archivo:** [requirements.txt](requirements.txt), [requirements-notebook.txt](requirements-notebook.txt)
- **Ubicación:** todo el archivo (`pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit` — sin `==`).
- **Descripción:** ningún paquete está fijado a una versión. Los `.joblib` son objetos **pickle de scikit-learn**: cargar un modelo entrenado con una versión de sklearn distinta a la de entrenamiento puede **fallar, emitir `InconsistentVersionWarning` o cambiar el comportamiento silenciosamente**.
- **Impacto:** un evaluador que clone el repo e instale dependencias frescas puede obtener un entorno que **no carga los modelos** o produce resultados distintos. La "reproducibilidad" que el proyecto presume no está garantizada.
- **Severidad:** Alta (producción/reproducibilidad), Media (defensa).
- **Solución:** congelar versiones (`pip freeze > requirements.txt`) o al menos fijar `scikit-learn==X.Y.Z`. Añadir la versión de Python en el README.

### 1.2 🟠 ALTO — El modelo de producción NO es el modelo tuneado documentado
- **Archivo:** [src/train.py](src/train.py) (líneas 42–54) vs [docs/document.md](docs/document.md) (§5) vs [notebooks/01_...ipynb](notebooks/01_housing_prices_proyecto_final.ipynb) (celda 37).
- **Evidencia (verificada):** `models/random_forest.joblib` carga con `n_estimators=300, max_depth=None`. Es decir, el RF de producción es el **base sin GridSearchCV**. El notebook sí tunea (`max_depth ∈ {None,10,20}`, etc.), pero **ese resultado nunca se persiste**; `src/train.py` entrena un RF con parámetros fijos.
- **Impacto:** la app y el CLI sirven un modelo **distinto** del que se analiza en el documento técnico. `max_depth=None` permite árboles a profundidad completa → **mayor sobreajuste** (el propio documento reporta gap train-test del ~61% para RF). El tuning que se defiende oralmente no está en el artefacto entregado.
- **Severidad:** Alta.
- **Solución:** que `src/train.py` ejecute GridSearchCV (o cargue los `best_params_` del notebook) y persista el modelo tuneado; regenerar `metrics_comparison.csv` con ese modelo.

### 1.3 🟡 MEDIO — Métricas publicadas inconsistentes con la documentación
- **Archivo:** [reports/metrics_comparison.csv](reports/metrics_comparison.csv) vs [docs/document.md](docs/document.md).
- **Evidencia:** CSV → RF `RMSE=27402.73, R²=0.9021`; LR `RMSE=32001.25, R²=0.8665`. Documento → RF `RMSE≈28253, R²≈0.896`; LR `RMSE≈31947`. **Ningún par coincide.**
- **Impacto:** un tribunal que cruce ambos documentos detectará la discrepancia y cuestionará la trazabilidad. Además el README fue editado para "corregir formato" del CSV (ver historial git), señal de que el archivo se manipuló manualmente.
- **Severidad:** Media (defensa).
- **Solución:** generar todas las métricas desde un único script reproducible y citarlas idénticas en todos lados.

### 1.4 🟡 MEDIO — Feature muerta: `has_masonry`
- **Archivo:** [src/preprocessing.py](src/preprocessing.py) (líneas 127–130).
- **Evidencia (verificada):** `has_masonry` se deriva de `MasVnrArea`, que **no está entre las 22 features**. En el modelo entrenado su **importancia es exactamente `0.0`**. Es una columna constante (siempre 0).
- **Impacto:** ruido conceptual; ocupa una columna en la matriz (64 features de salida) sin aportar señal. Riesgo de pregunta incómoda: *"¿por qué tu modelo tiene una variable que siempre vale lo mismo?"*.
- **Severidad:** Media (defensa), Baja (técnica).
- **Solución:** eliminar el flag o reincorporar `MasVnrArea` a las features para que tenga sentido.

### 1.5 🟢 BAJO — `StandardScaler` aplicado innecesariamente al Random Forest
- **Archivo:** [src/train.py](src/train.py) (línea 44, usa `build_preprocessor()` con default `scale_numeric=True`).
- **Evidencia (verificada):** el pipeline numérico del RF de producción incluye `['imputer', 'scaler']`. Los árboles son **invariantes a transformaciones monótonas** → el escalado no cambia las predicciones, solo añade cómputo.
- **Impacto:** ninguno en resultados; inconsistencia con el notebook (que usa `scale_numeric=False` para RF) y trabajo desperdiciado.
- **Severidad:** Baja.
- **Solución:** pasar `scale_numeric=False` al construir el pipeline del RF, como en el notebook.

### 1.6 🟢 BAJO — Artefacto huérfano `models/preprocessor.joblib`
- **Archivo:** generado por [scripts/check_preprocessor.py](scripts/check_preprocessor.py).
- **Descripción:** existe un `preprocessor.joblib` (6.9 KB) que **ningún módulo de producción usa** (ni `train`, ni `predict`, ni la app). Es un subproducto de un script de verificación.
- **Impacto:** confusión sobre qué artefactos son canónicos.
- **Severidad:** Baja.
- **Solución:** eliminarlo o documentarlo como artefacto de diagnóstico.

### 1.7 🟢 BAJO — Ramas de `DomainImputer` que nunca se ejecutan en entrenamiento
- **Archivo:** [src/preprocessing.py](src/preprocessing.py) (manejo de `GarageType`, `GarageFinish`, `MasVnrType`).
- **Evidencia (verificada):** en `train.csv`, de las 22 features **solo `BsmtQual` tiene nulos** (37). `GarageArea`/`GarageCars` no tienen nulos. Las ramas de garaje/mampostería del imputador **no se activan** durante el entrenamiento (sí servirían para robustez en inferencia con columnas extra).
- **Impacto:** código que parece manejar casos que en este dataset no ocurren → puede dar falsa impresión de cobertura.
- **Severidad:** Baja (es defensivo, no incorrecto).

### 1.8 🟢 BAJO — Sin pruebas, sin CI, sin linter
- **Descripción:** no hay `tests/`, ni `pytest`, ni configuración de CI, ni `pyproject.toml`. Cero red de seguridad ante regresiones.
- **Severidad:** Baja (curso), Alta (producción).

---

# Sección 2 — Auditoría de Data Leakage

> **Veredicto general: el proyecto está LIMPIO de leakage grave.** El uso del `Pipeline` es correcto y es su mayor fortaleza. Solo hay observaciones menores.

| Vector de fuga | Estado | Evidencia |
|---|---|---|
| **Train/Test Split** | ✅ Correcto | `train_test_split` se ejecuta **antes** de cualquier `fit`; el test no toca el ajuste |
| **Pipeline / fit** | ✅ Correcto | Imputador, scaler, clipper, rare-grouper hacen `fit` solo con el train de cada contexto |
| **ColumnTransformer** | ✅ Correcto | Está dentro del pipeline; se reajusta por fold |
| **Escalado** | ✅ Correcto | `StandardScaler.fit` usa solo train (media/σ de train) |
| **Imputación** | ✅ Correcto | Mediana/moda calculadas en `fit` (train) |
| **Cross-validation** | ✅ Correcto | `cross_validate`/`GridSearchCV` reciben el **pipeline completo** → cada fold reajusta preprocesamiento. **Esto es lo que muchos proyectos hacen mal y aquí está bien.** |
| **GridSearchCV** | ✅ Correcto | Opera sobre el pipeline; sin fuga |

### Observaciones de leakage **leve / metodológico** (no invalidan resultados)

**2.1 🟡 Leakage humano en EDA / selección de variables.**
El EDA (correlaciones, conteo de outliers, mediana por vecindario) y la **selección de las 22 features** se hicieron mirando el **dataset completo** (`df`), incluyendo las filas que luego serán test. Las decisiones (qué variables conservar, dónde poner los umbrales de clipping conceptuales) están **informadas por el test**.
- **Riesgo:** optimismo leve y no cuantificable. Es el típico "researcher degrees of freedom".
- **Cómo corregir:** decidir features y umbrales **solo con el train**, o usar un conjunto de validación separado para esas decisiones. En la práctica académica es aceptable, pero un auditor estricto lo señala.

**2.2 🟢 Defaults de inferencia desde el train completo.**
`predict.py` y la app calculan mediana/moda desde **todo** `train.csv` (incluye el 20% de test). No es fuga hacia el modelo (el modelo no se reentrena), pero los "valores típicos" que rellenan features faltantes incorporan el test.
- **Riesgo:** despreciable.

**2.3 🟢 `feature_diagnostics.py` ajusta un RF sobre datos completos.**
Solo para reportes de importancia/correlación; **no alimenta el entrenamiento**. Sin impacto en el modelo servido.

---

# Sección 3 — Validación Experimental

### ¿La evaluación es confiable?
**Parcialmente.** La metodología (split 80/20 + KFold=5 + GridSearchCV con scoring RMSE) es **correcta y estándar**. Pero:

- **3.1 🟡 Las métricas de producción son un punto único sin intervalo.** `src/train.py` reporta el RMSE de **un solo split**, sin desviación estándar. El CSV publicado no transmite la incertidumbre. El notebook sí hace CV (mejor), pero ese resultado no es el que se publica.
- **3.2 🟡 Selección de "ganador" usando el test.** El modelo ganador se decide por `test_rmse`. Con solo 2 modelos el riesgo de overfitting de selección es bajo, pero formalmente el test se usa para **comparar y elegir**, no solo para reportar. Lo riguroso sería elegir por CV y reportar el test una sola vez.

### ¿Existe optimismo artificial?
Leve, por 2.1 (decisiones informadas por todo el dataset) y 3.2. No es grave.

### ¿Existe sobreajuste?
**Sí, en el Random Forest, y está mal mitigado en producción.** El documento reporta gap train-test del **~61%** (RMSE train ≈11.043 vs test ≈28.253). El modelo de producción tiene `max_depth=None` → **aún más libre para memorizar**. El tuning que controlaría esto no está en el artefacto entregado (ver 1.2). La defensa de "lo controlo con CV y max_depth" **no aplica al modelo realmente servido**.

### ¿Existe subajuste?
La Regresión Lineal está cerca del subajuste (no modela no linealidades), pero es esperado y sirve de línea base. Aceptable.

### ¿Las métricas son consistentes?
**No** (ver 1.3). Es el punto más débil de esta sección.

---

# Sección 4 — Auditoría de Random Forest

| Aspecto | Hallazgo |
|---|---|
| **Hiperparámetros (producción)** | 🟠 `n_estimators=300, max_depth=None, min_samples_leaf=1` → configuración de **máxima varianza** (árboles a profundidad completa, hojas de 1 muestra). Es el peor caso para sobreajuste con n=1460 |
| **Hiperparámetros (notebook)** | ✅ La grilla es razonable, pero su resultado no se persiste |
| **Entrenamiento** | ✅ `random_state=42`, `n_jobs=-1` correcto |
| **`StandardScaler` redundante** | 🟢 Innecesario (1.5) |
| **Importancia de variables** | 🟡 Usa **importancia por impureza** (Gini/MDI), que está **sesgada hacia variables de alta cardinalidad** (favorece numéricas continuas y `Neighborhood` con muchas categorías). Sería más fiable la **importancia por permutación** |
| **Interpretabilidad** | ✅ La app agrega importancias one-hot por variable original — buena práctica |

**¿Está correctamente configurado?** Funciona, pero **no óptimamente en producción** (sin profundidad limitada). **¿Parámetros innecesarios?** El scaler. **¿Mejoras?** Persistir el modelo tuneado; usar `permutation_importance`; limitar `max_depth`/`min_samples_leaf`.

---

# Sección 5 — Auditoría del Pipeline (componente por componente)

### DomainImputer
- **Qué hace:** reglas de dominio para nulos estructurales + flags `has_garage/has_bsmt/has_masonry`.
- **¿Correcto?** Sí, lógicamente. **Errores:** `has_masonry` muerta (1.4); ramas inactivas en train (1.7).
- **Mejora:** usar `.loc` consistentemente (ya lo hace); eliminar flags constantes.
- ⚠️ **Sutileza verificada y DESCARTADA:** se sospechó que el clipping posterior subiría a >0 los `TotalBsmtSF=0` (sin sótano), contradiciendo `has_bsmt`. **No ocurre:** el percentil 1 de `TotalBsmtSF` y `GarageArea` es **0** (hay 2.5% y 5.5% de ceros), así que el límite inferior del clip es 0 y los ceros se preservan. **No hay bug aquí.**

### RareCategoryGrouper
- **Qué hace:** agrupa categorías <1% en "Other", aprendiendo en `fit` (solo train).
- **¿Correcto?** Sí. **Posible problema:** umbral fijo 1% sin justificación empírica; en columnas con pocas categorías (ej. `ExterQual`) puede no activarse nunca.
- **Mejora:** validar el umbral; documentar qué categorías agrupa realmente.

### QuantileClipper
- **Qué hace:** winsoriza al p1/p99, límites de `fit` (train).
- **¿Correcto?** Sí, sin fuga. **Observación:** winsorizar **antes** del split conceptual no aplica (está en el pipeline), bien. Recorta también variables ya tratadas por reglas de dominio, pero es inocuo (verificado).
- **Mejora:** clip asimétrico (solo cola superior) para áreas, ya que el p1 suele ser estructural.

### ColumnTransformer
- **¿Correcto?** Sí. Salida verificada: **64 columnas**. `remainder="drop"` evita columnas no declaradas. Bien.

### One-Hot Encoding
- **¿Correcto?** Sí. `handle_unknown="ignore"` tolera categorías nuevas en inferencia. En `src` usa `sparse_output=False` (denso) → mayor uso de memoria pero compatible con todos los modelos.
- **Mejora menor:** `drop="first"` reduciría colinealidad de las dummies para la Regresión Lineal (no afecta al RF).

### Escalado
- Correcto en LR; **redundante en RF** (1.5).

---

# Sección 6 — Auditoría Arquitectónica

### Fortalezas
- **`app/` con arquitectura en capas** (repositorio, servicios, validación, analítica, UI, catálogo) — separación de responsabilidades **excelente** para un proyecto de curso.
- `src/` desacoplado de la app; manifest JSON como fuente única de features.

### Debilidades

**6.1 🟡 Violación de DRY — código duplicado notebook ↔ src.**
`DomainImputer`, `RareCategoryGrouper`, `QuantileClipper`, `build_preprocessor`, métricas: **dos implementaciones divergentes** (ver Sección 7). Cualquier cambio debe hacerse dos veces; ya divergieron.

**6.2 🟡 Sobre-ingeniería en la app.**
`InputPreparationService` incluye `apply_preset` (Casa compacta/familiar/premium) y `build_ab_scenario` (simulador A/B con deltas de área/calidad/garaje) que **no se usan en `streamlit_app.py`**. Es código muerto de funcionalidad: mantenido pero no cableado a la UI actual.
- **Solución:** conectar esas features o eliminarlas.

**6.3 🟢 Acoplamiento a Streamlit en la capa de servicios.**
`services.py` decora con `@st.cache_data`/`@st.cache_resource` → la capa de "datos" depende de Streamlit. Dificulta testear `DataRepository` fuera de la app.
- **Solución:** separar carga pura del cacheo de UI.

**6.4 🟢 Configuración dispersa.**
Las 22 features están definidas en **3 lugares** (`features.json`, `FEATURE_COLUMNS` en `src`, y `NUMERIC_FEATURES/CATEGORICAL_FEATURES` en el notebook). Riesgo de desincronización.

**SOLID:** la app respeta SRP razonablemente. La principal violación es **DRY** (no SOLID estricto) entre notebook y src.

---

# Sección 7 — Notebook vs Producción (comparación exhaustiva)

| Dimensión | Notebook | `src/` producción | Impacto / Riesgo |
|---|---|---|---|
| **DomainImputer** | Solo `BsmtQual` + garaje numérico; flags `has_garage`, `has_bsmt` | + `MasVnrType`, `GarageType`, `GarageFinish`, `ensure_object_dtype`; flag extra `has_masonry` | Producción es superconjunto; genera la feature muerta |
| **Flags** | 2 | 3 (`has_masonry` constante) | 1.4 |
| **OneHotEncoder** | sparse por defecto | `sparse_output=False` | Inocuo |
| **Escalado RF** | `scale_numeric=False` ✅ | `scale_numeric=True` (scaler en RF) ❌ | Cómputo desperdiciado (1.5) |
| **Winsorización** | Siempre | Parametrizable (`winsorize=True`) | Equivalente por defecto |
| **Tuning** | **GridSearchCV** | **Ninguno** | **El artefacto servido NO está tuneado (1.2)** |
| **Modelo persistido** | No persiste el tuneado | Persiste el base | Lo defendido ≠ lo entregado |
| **Métricas** | RF RMSE≈28253, R²≈0.896 | CSV RF RMSE=27402, R²=0.902 | Inconsistencia (1.3) |
| **Features** | Idénticas (22) | Idénticas (22) | ✅ Consistente |

### Cómo defender cada diferencia ante el tribunal
- **Tuning ausente en producción:** *"El notebook documenta el experimento completo con tuning; `src/train.py` entrena una versión base reproducible. Reconozco que lo ideal es persistir el modelo tuneado y es mi primera corrección pendiente."* (Honestidad > inventar.)
- **Métricas distintas:** *"Provienen de pipelines distintos —base vs tuneado—. Unificaré la fuente de métricas."* (Nota posterior: en validación cruzada RF y LR son **estadísticamente equivalentes**; RF solo es el mejor estimador puntual en el test. Ver `VALIDACION_CIENTIFICA_FINAL.md`.)
- **`has_masonry`:** *"Flag defensivo de una versión con `MasVnrArea`; hoy es constante y el modelo le asigna importancia cero. Lo eliminaré."*
- **Código duplicado:** *"El notebook es autocontenido por requisito; en refactor importaré desde `src` para eliminar la duplicación."*

---

# Sección 8 — Robustez de la App (intentos de romperla)

| Escenario de ataque | Resultado | Severidad |
|---|---|---|
| **Modelos ausentes** | ✅ Manejado: `st.error` + `st.stop()` |  — |
| **`train.csv` ausente** | 🟡 `load_reference_data` devuelve `None`; la app usa fallbacks, pero los sliders pierden rangos reales y los percentiles desaparecen. Degradación silenciosa | Bajo |
| **`metrics_comparison.csv` ausente** | ✅ Maneja `None`; consenso cae a promedio simple, rango a ±7% | — |
| **Input categórico inválido** | ✅ `InputValidator` restaura default con nota | — |
| **Input numérico fuera de rango** | 🟡 Se **clipa silenciosamente** a p1/p99 del dataset. Una mansión real (GrLivArea 6000) se recorta a ~2800 → **subestimación oculta** sin avisar claramente al usuario | Medio (UX/credibilidad) |
| **Doble clipping** | 🟡 El input se clipa en `sanitize_values` **y** otra vez en `QuantileClipper` dentro del pipeline. Redundante y refuerza la subestimación de extremos | Bajo |
| **Inyección HTML** | 🟢 `render_*` usan `unsafe_allow_html=True` con f-strings. Los valores inyectados son numéricos o nombres de barrio del dataset (no entrada libre del usuario) → riesgo XSS bajo, pero la práctica es peligrosa si se extiende a texto libre | Bajo |
| **Predicción negativa** | 🟡 `LinearRegression` puede predecir precios negativos con inputs extremos; el rango usa `max(pred - margin, 0)` pero la predicción cruda del lineal no tiene piso | Bajo |

**Conclusión:** la app es **robusta ante errores de operación** (modelos/datos faltantes, inputs inválidos), pero **engaña en los extremos**: clipa sin transparencia y subestima propiedades de lujo, justo donde el modelo ya es débil. Esto es coherente con la limitación del modelo, pero la UI no lo comunica con suficiente claridad.

---

# Sección 9 — Críticas de Tribunal (modo "reprobar el proyecto")

**9.1 — "Su documento dice que tuneó el modelo, pero el modelo que entrega tiene `max_depth=None`. ¿Me está mostrando resultados de un modelo que no es el que corre la app?"**
- *Por qué la harían:* es la grieta más grande; cruzar CSV, doc y `.joblib` lo revela.
- *Cómo responder:* reconocerlo sin rodeos (ver 1.2). No defender lo indefendible; mostrar que se entiende la consecuencia (más sobreajuste) y el plan de corrección.

**9.2 — "Eliminó `BsmtFinSF1` y `BsmtUnfSF` por ser redundantes con `TotalBsmtSF`, pero conservó `1stFlrSF` y `2ndFlrSF` que suman `GrLivArea` (correlación 0.996). ¿No es contradictorio?"**
- *Por qué:* es una inconsistencia real en el criterio de selección, verificada.
- *Cómo responder:* *"Es una observación válida. Mantuve `1st/2ndFlrSF` porque aportan la **distribución por piso** (no solo el total), pero reconozco la multicolinealidad; para la Regresión Lineal convendría consolidarlas o usar `drop='first'`. Para el Random Forest no afecta."*

**9.3 — "Su Random Forest memoriza (gap 61%). ¿Por qué debería confiar en él?"**
- *Cómo responder:* *"La selección se respalda con **validación cruzada**, no solo con el test. En CV el RF y el lineal son equivalentes (p≈0.13) y el gap del 60% es la principal limitación del RF. Aun así, RF es el **modelo final** por su mejor desempeño en test; `Regresión Lineal + log1p` (gap 1.8%) queda documentada como **alternativa más estable, no seleccionada como modelo operativo**. Ver `VALIDACION_CIENTIFICA_FINAL.md`."*

**9.4 — "Su app recorta mis datos sin avisarme. Si vendo una mansión, ¿me da un precio falso?"**
- *Cómo responder:* *"El sistema es fiable en el rango medio (mayor densidad de datos) y advierte que no debe usarse como tasación oficial. En lujo subestima por falta de ejemplos; debería comunicarlo más explícitamente — punto de mejora aceptado."*

**9.5 — "¿Cómo sé que sus números son reproducibles si no fijó versiones de librerías?"**
- *Cómo responder:* reconocer (1.1) y comprometer `pip freeze`.

**9.6 — "Tiene una variable que siempre vale cero. ¿Revisó su propio modelo?"**
- *Cómo responder:* explicar `has_masonry` (1.4) y que su importancia es 0; eliminarla.

**9.7 — "¿Por qué su métrica principal es RMSE y no MAE, si tiene outliers que sabe que distorsionan el RMSE?"**
- *Cómo responder:* *"Precisamente por eso: en tasación, los errores grandes son los más costosos, y quiero penalizarlos. Reporto MAE y MAPE en paralelo para el error típico robusto."*

---

# Sección 10 — Priorización

| # | Hallazgo | Severidad | Riesgo Defensa | Riesgo Técnico | Prioridad |
|---|---|---|---|---|---|
| 1.2 | Modelo de producción sin tunear ≠ documentado | 🟠 Alto | **Muy alto** | Alto (sobreajuste) | **P0** |
| 1.3 | Métricas CSV ≠ documento | 🟡 Medio | Alto | Bajo | **P0** |
| 1.1 | Sin pins de dependencias | 🟠 Alto | Medio | Alto | **P1** |
| 9.2 | Redundancia `1st+2ndFlr`≈`GrLivArea` mantenida | 🟡 Medio | Alto | Medio (LR) | **P1** |
| 1.4 | Feature muerta `has_masonry` | 🟡 Medio | Medio | Bajo | **P1** |
| 3.x | Sobreajuste RF no mitigado en producción | 🟠 Alto | Medio | Alto | **P1** |
| 8 | App clipa/subestima extremos sin transparencia | 🟡 Medio | Medio | Bajo | **P2** |
| 6.1 | Código duplicado notebook↔src | 🟡 Medio | Bajo | Medio | **P2** |
| 1.5 | Scaler redundante en RF | 🟢 Bajo | Bajo | Bajo | **P2** |
| 6.2 | Código muerto (`apply_preset`, A/B) | 🟢 Bajo | Bajo | Bajo | **P3** |
| 1.8 | Sin tests / CI | 🟢 Bajo | Bajo | Alto (prod) | **P3** |
| 2.1 | Selección/EDA sobre dataset completo | 🟡 Medio | Bajo | Bajo | **P3** |
| 1.6 | Artefacto huérfano `preprocessor.joblib` | 🟢 Bajo | Bajo | Bajo | **P3** |

---

# Sección 11 — Veredicto Profesional

### 1. ¿Aprobarías este proyecto en una defensa universitaria?
**Sí, lo aprobaría** — con holgura. La metodología es sólida, el control de data leakage es correcto (mejor que la mayoría de proyectos de curso), la arquitectura de la app es notable y el análisis crítico es maduro. Los defectos son de **rigor y consistencia**, no de comprensión.

### 2. ¿Qué nota le pondrías?
**85–88 / 100 (B+ / "Muy Bueno").**
- Pierde puntos por: desconexión modelo documentado vs servido (1.2/1.3), reproducibilidad no garantizada (1.1), redundancia contradictoria (9.2) y ausencia de pruebas.
- Gana puntos por: pipeline sin fuga, EDA con decisiones justificadas, evaluación con residuos/importancia, app profesional y análisis ético.
- Llegaría a **95+** si se corrigen los P0/P1.

### 3. ¿Qué errores corregirías ANTES de la defensa? (P0–P1, ~2 horas)
1. **Unificar métricas:** regenerar `metrics_comparison.csv` con un único script y citarlas idénticas en doc/README (1.3).
2. **Decidir y persistir UN modelo:** o bien tunear en `src/train.py`, o bien declarar explícitamente que el modelo servido es el base y alinear el documento (1.2).
3. **Preparar la respuesta honesta** a "modelo documentado ≠ servido" y a "redundancia de áreas" (9.1, 9.2).
4. **Eliminar `has_masonry`** o justificar su presencia (1.4).
5. Añadir versiones a `requirements.txt` (1.1).

### 4. ¿Qué errores corregirías ANTES de producción? (adicionales)
1. **Pins de dependencias + versión de Python** y validación de versión de sklearn al cargar `.joblib`.
2. **Persistir el modelo tuneado** y limitar `max_depth`/`min_samples_leaf` (sobreajuste).
3. **Tests** (pipeline, validación de inputs, carga de modelos) + CI.
4. **Transparencia en extremos:** avisar al usuario cuando su input fue clipado; poner piso a la predicción lineal.
5. **`permutation_importance`** en vez de MDI para la importancia mostrada.
6. **Eliminar duplicación** notebook↔src y código muerto de servicios.
7. **Intervalos de predicción reales** (cuantílicos) en lugar de ±MAE heurístico.

### 5. ¿Cuál es el hallazgo MÁS IMPORTANTE de todo el repositorio?
> **🔴 El modelo que se defiende no es el modelo que se entrega (1.2 + 1.3).**
> El documento técnico presume un Random Forest tuneado con GridSearchCV (RMSE≈28253, R²≈0.896), pero `models/random_forest.joblib` es el modelo **base sin tunear** (`max_depth=None`) y `metrics_comparison.csv` publica **otros números** (RMSE=27402, R²=0.902). Es una **brecha de trazabilidad entre experimento, artefacto y documentación** que un tribunal exigente puede explotar para cuestionar toda la evaluación. Es barato de arreglar y es lo primero que corregiría.

---

## Nota de cierre del auditor

Este es un **buen proyecto** debilitado por **descuidos de consistencia y reproducibilidad**, no por errores conceptuales. El autor entiende ML; lo que falla es la disciplina de MLOps (un solo source of truth, versionado, tests) y la coherencia entre lo experimentado, lo entregado y lo documentado. Corregir los cuatro P0/P1 lo convierte en un trabajo sobresaliente y a prueba de tribunal.

*Fin de la auditoría.*
