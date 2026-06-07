# 🛠️ CORRECCIONES REALIZADAS

> ⚠️ **ACTUALIZACIÓN POSTERIOR.** Este documento cubre la remediación de **consistencia y
> reproducibilidad**. Una **validación científica posterior** confirmó que en validación
> cruzada el Random Forest y la Regresión Lineal son **estadísticamente equivalentes**
> (p≈0.13) y que `log1p` mejora la **estabilidad** del modelo lineal (sobreajuste 1.8%).
> **POSTURA OFICIAL DE SELECCIÓN:** Random Forest es el **modelo final del sistema**
> (mejor desempeño predictivo en test). `Regresión Lineal + log1p` se documenta como
> **alternativa más estable y con menor sobreajuste, no seleccionada como modelo
> operativo** (reproducible con `python -m src.train --log-target`). Conclusión final en
> `VALIDACION_CIENTIFICA_FINAL.md` y `ALINEAMIENTO_FINAL_CON_RETROALIMENTACION_PROFESOR.md`.

> Remediación de los hallazgos de `AUDITORIA_TECNICA_COMPLETA.md`.
> **Fuente de verdad oficial decidida:** `src/` (producción). El notebook se mantiene
> como evidencia académica autocontenida; se verificó que **ya coincidía** con la
> conducta correcta (sin `has_masonry`, Random Forest sin escalado), de modo que las
> correcciones **alinean `src/` hacia el notebook** y ambos convergen.
> **Todas las cifras de este documento fueron regeneradas y verificadas ejecutando el código.**

---

## Resumen de cambios

| # | Corrección | Severidad original | Archivos | Estado |
|---|---|---|---|---|
| 1 | Eliminada la feature muerta `has_masonry` | 🟡 Medio (P1) | `src/preprocessing.py` | ✅ |
| 2 | Eliminado código inalcanzable del `DomainImputer` | 🟢 Bajo (P3) | `src/preprocessing.py` | ✅ |
| 3 | `StandardScaler` quitado del Random Forest | 🟢 Bajo (P2) | `src/train.py` | ✅ |
| 4 | Métricas unificadas (CSV = documento = app) | 🟡 Medio (P0) | `reports/metrics_comparison.csv`, `docs/document.md` | ✅ |
| 5 | Tuning reproducible desde producción (`--tune`) | 🟠 Alto (P0) | `src/train.py`, `README.md` | ✅ |
| 6 | Dependencias fijadas (pins) + Python 3.11 | 🟠 Alto (P1) | `requirements.txt`, `README.md` | ✅ |
| 7 | `requirements-notebook` corregido (faltaba `ipywidgets`; sobraban libs) | 🟡 Medio | `requirements-notebook.txt` | ✅ |

---

## Corrección 1 — Feature muerta `has_masonry`

- **Problema.** `DomainImputer` creaba el flag `has_masonry` a partir de `MasVnrArea`,
  variable que **no pertenece al set de 22 features**. En el modelo entrenado su
  importancia era **exactamente 0.0**: una columna constante sin efecto.
- **Impacto.** Ruido conceptual y una columna inútil en la matriz de entrada (64ª
  feature). Riesgo de pregunta de tribunal: *"¿por qué tu modelo tiene una variable
  que siempre vale lo mismo?"*.
- **Solución.** Se eliminó la creación de `has_masonry` en `DomainImputer.transform`
  y su inclusión en el bloque numérico de `build_preprocessor`. La matriz de
  entrada pasó de **64 → 63 features** (verificado: `n_features_in_ = 63`).
- **Archivos modificados.** `src/preprocessing.py` (clase `DomainImputer`, función
  `build_preprocessor`).
- **Justificación técnica.** Una variable constante no aporta información: no genera
  divisiones en los árboles ni varianza para la regresión. Eliminarla **no cambia
  las predicciones** (verificado: la Regresión Lineal mantiene métricas idénticas y
  el Random Forest varía solo por el reordenamiento de columnas en su RNG, R²
  0.9021 → 0.9018, diferencia trivial), pero elimina deuda técnica y una pregunta
  incómoda.

---

## Corrección 2 — Código inalcanzable en `DomainImputer`

- **Problema.** El imputador manejaba `MasVnrType`, `GarageType` y `GarageFinish`
  (etiquetas `"NoMasonry"`/`"NoGarage"`), columnas que **no están en las 22 features**
  ni se generan en inferencia → ramas que **nunca se ejecutan**.
- **Impacto.** Falsa impresión de cobertura; divergencia con el `DomainImputer` del
  notebook (más simple).
- **Solución.** Se redujo `DomainImputer` a las reglas que sí operan sobre features
  reales: `BsmtQual` NA → `"NoBasement"` con `TotalBsmtSF=0`, coerción numérica de
  `GarageArea`/`GarageCars`, y los flags `has_garage` / `has_bsmt`. Queda
  funcionalmente equivalente al imputador del notebook.
- **Archivos modificados.** `src/preprocessing.py`.
- **Justificación técnica.** El imputador ahora refleja exactamente lo que el dataset
  necesita (solo `BsmtQual` tiene nulos entre las 22 features; `GarageArea`/`GarageCars`
  no tienen nulos en `train.csv`). Se conservan las guardas `if col in X.columns`
  para robustez si el manifest cambia, sin ramas muertas.

---

## Corrección 3 — `StandardScaler` redundante en el Random Forest

- **Problema.** `src/train.py` construía el pipeline del Random Forest con
  `build_preprocessor()` por defecto (`scale_numeric=True`), aplicando un
  `StandardScaler` a un modelo de árboles.
- **Impacto.** Cómputo inútil e inconsistencia con el notebook (que usa
  `scale_numeric=False` para el RF). Sin efecto en las predicciones.
- **Solución.** El RF ahora se construye con `build_preprocessor(scale_numeric=False)`;
  la Regresión Lineal mantiene `scale_numeric=True`.
- **Archivos modificados.** `src/train.py` (`build_models`).
- **Justificación técnica.** Los árboles de decisión parten por umbrales y son
  **invariantes a transformaciones monótonas** como la estandarización; escalar no
  cambia las divisiones. La Regresión Lineal **sí** requiere escalado porque es
  sensible a la magnitud de las variables.

---

## Corrección 4 — Métricas inconsistentes (CSV ↔ documento ↔ app)

- **Problema.** `reports/metrics_comparison.csv` (RF RMSE 27.402, R² 0.902) **no
  coincidía** con `docs/document.md` (RF RMSE ≈ 28.253, R² ≈ 0.896), porque el
  documento citaba el modelo **tuneado del notebook** mientras el CSV reflejaba el
  modelo **base servido**.
- **Impacto.** Brecha de trazabilidad; un tribunal que cruzara ambos detectaría la
  contradicción y cuestionaría toda la evaluación.
- **Solución.** Se estableció `reports/metrics_comparison.csv` (generado por
  `src/train.py`) como **fuente única de verdad**, ya consumida por la app. Se
  regeneró tras las correcciones y se actualizó `docs/document.md` para citar
  **exactamente** esas cifras:

  | Modelo | MAE | RMSE | R² |
  |---|---|---|---|
  | Random Forest (servido) | 17.198 | 27.449 | 0.9018 |
  | Regresión Lineal | 18.171 | 32.001 | 0.8665 |

  También se actualizó la tabla de diagnóstico de sobreajuste (§5.3) con los valores
  reales del modelo base: LR gap 12.1% (28.137 → 32.001), RF gap 59.9% (11.012 →
  27.449), y las conclusiones (§8).
- **Archivos modificados.** `reports/metrics_comparison.csv`, `docs/document.md`.
- **Justificación técnica.** El único conjunto de métricas defendible es el del
  **artefacto que realmente corre** (la app y el CLI cargan `models/*.joblib`). El
  notebook queda como exploración metodológica; el documento ahora aclara que las
  métricas oficiales son las del modelo servido y que el tuning es reproducible
  (Corrección 5).

---

## Corrección 5 — Tuning reproducible desde producción (`--tune`)

- **Problema.** El `GridSearchCV` solo existía en el notebook; su resultado nunca se
  persistía. El modelo servido era el base sin ajuste (`max_depth=None`), pero el
  documento presumía un modelo tuneado → *"lo que se defiende no es lo que se entrega"*
  (hallazgo #1 de la auditoría).
- **Impacto.** Imposible regenerar el modelo ajustado desde el código de producción;
  inconsistencia entre experimento, artefacto y documentación.
- **Solución.** Se añadió la bandera `python -m src.train --tune`, que ejecuta el
  **mismo** GridSearchCV del notebook (grillas idénticas, KFold=5, scoring RMSE) y
  **persiste el modelo ajustado**. Por defecto (sin la bandera) sigue entrenando la
  configuración base → **sin regresión** ni penalización de tiempo en el flujo normal.
  El comportamiento queda documentado en el `README.md` y en `docs/document.md`.
- **Archivos modificados.** `src/train.py` (`PARAM_GRIDS`, `train_and_evaluate(tune=...)`,
  `--tune` en `parse_args`/`main`), `README.md`.
- **Justificación técnica.** El tuning deja de ser exclusivo del notebook y pasa a ser
  un camino **reproducible y versionado** de producción. La decisión de servir el
  modelo base por defecto prioriza determinismo y velocidad; la decisión de exponer
  `--tune` cierra la brecha de trazabilidad sin imponer el costo del GridSearch a
  cada entrenamiento.

---

## Corrección 6 — Reproducibilidad: dependencias fijadas

- **Problema.** `requirements.txt` no fijaba ninguna versión. Los `.joblib` son
  pickles de scikit-learn: cargarlos con otra versión puede fallar o cambiar el
  comportamiento silenciosamente.
- **Impacto.** La reproducibilidad que el proyecto presume no estaba garantizada para
  quien clonara el repo.
- **Solución.** Se fijaron las versiones probadas (`pandas==2.3.3`, `numpy==2.4.3`,
  `scikit-learn==1.8.0`, `joblib==1.5.3`, `streamlit==1.57.0`) y se documentó
  **Python 3.11** en `README.md` y en el encabezado del `requirements.txt`.
- **Archivos modificados.** `requirements.txt`, `README.md`.
- **Justificación técnica.** Fijar `scikit-learn` es lo más crítico porque garantiza
  que el modelo serializado se cargue y prediga igual que cuando se entrenó.

---

## Corrección 7 — `requirements-notebook.txt` corregido

- **Problema.** (a) Faltaba `ipywidgets`, **requerido** por el sistema interactivo del
  notebook (celda de widgets) → el notebook no corría con un entorno limpio. (b)
  Listaba `xgboost`, `lightgbm`, `catboost` y `plotly`, que el notebook **no importa**
  (y `lightgbm`/`catboost` ni estaban instaladas) → dependencias innecesarias.
- **Impacto.** Entorno de notebook roto (falta `ipywidgets`) e instalación de
  paquetes pesados sin uso.
- **Solución.** Se dejó solo lo que el notebook importa realmente, con versiones
  fijadas: `matplotlib`, `seaborn`, `jupyter`, `notebook`, `ipywidgets`.
- **Archivos modificados.** `requirements-notebook.txt`.
- **Justificación técnica.** Un archivo de dependencias debe reflejar los imports
  reales: ni de menos (rompe la ejecución) ni de más (instalación lenta y superficie
  de fallo mayor).

---

## Verificación de NO regresión (ejecutada)

| Prueba | Resultado |
|---|---|
| `python -m src.train` (reentrenamiento base) | ✅ Modelos + CSV regenerados |
| `python -m src.predict --tamano 120 --ubicacion NAmes --habitaciones 3` | ✅ LR 124.511 · RF 142.743 |
| Inferencia por la ruta de la app (frame 22 cols + ambos modelos) | ✅ Predicciones coherentes |
| `PredictionAnalytics.feature_importances` (app) | ✅ OverallQual 56.5%, GrLivArea 13.3% (coherente con EDA) |
| `weighted_consensus` (app) | ✅ Funciona |
| Ausencia de `has_masonry` en el pipeline entrenado | ✅ Confirmada (63 features) |
| Regresión Lineal: métricas tras los cambios | ✅ **Idénticas** (la feature eliminada era constante) |

**Conclusión:** todas las correcciones se aplicaron al repositorio real sin romper
el CLI, la app ni la lógica de analítica. El comportamiento por defecto se preserva.

---

## Convergencia notebook ↔ producción (estado final)

| Aspecto | Notebook | `src/` (antes) | `src/` (ahora) |
|---|---|---|---|
| Flags del `DomainImputer` | `has_garage`, `has_bsmt` | + `has_masonry` ❌ | `has_garage`, `has_bsmt` ✅ |
| Escalado del Random Forest | No | Sí ❌ | No ✅ |
| Ramas inalcanzables | No | Sí ❌ | No ✅ |
| Features de entrada al modelo | 63 | 64 | 63 ✅ |
| Tuning reproducible | Solo notebook | No en `src` | `--tune` ✅ |

---

## Notas sobre los documentos de análisis previos

`AUDITORIA_TECNICA_COMPLETA.md` y `MANUAL_MAESTRO_DEFENSA.md` se conservan como
**fotografías del estado anterior** (auditoría y guía de estudio). Tras estas
correcciones quedan **superados** los hallazgos: 1.2 (modelo servido vs documentado),
1.3 (métricas inconsistentes), 1.4 (`has_masonry`), 1.5 (scaler en RF) y 1.1
(dependencias sin pin). El §18 del manual (inconsistencias) describe problemas hoy
resueltos; úsese como contexto histórico, no como estado actual.

---

## Pendientes deliberadamente NO aplicados (fuera de alcance / requieren decisión)

- **Persistir por defecto el modelo tuneado.** Se dejó el base por defecto para no
  introducir regresión de tiempo/determinismo; `--tune` queda disponible.
- **Tests automatizados / CI.** Recomendado para producción; no forma parte de esta
  remediación de consistencia.
- **`permutation_importance`** en lugar de importancia por impureza (MDI) en la app.
- **Transparencia de clipping en la app** y piso a la predicción lineal.
- **Eliminar duplicación de clases notebook↔src** importando desde `src` (rompería la
  autocontención exigida al notebook).

*Fin del informe de correcciones.*
