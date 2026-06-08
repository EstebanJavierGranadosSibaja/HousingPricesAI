# Reporte de Cambios — Proyecto Final IA

Estado real del proyecto a 2026-06-08. Documenta todos los cambios aplicados desde la
entrega inicial. Sin modificaciones de modelos base, CSVs ni datos crudos.

---

## Estado actual de los entregables

| Entregable | Estado |
|---|---|
| `notebooks/01_housing_prices_proyecto_final.ipynb` | ✅ Completo, re-ejecutado, determinista |
| `docs/document.md` | ✅ Alineado con notebook y producción |
| `app/streamlit_app.py` | ✅ Funcional — 4 pestañas operativas |
| `reports/metrics_comparison.csv` | ✅ Fuente de verdad (no modificada) |
| `reports/ablation_study.csv` | ✅ Fuente de verdad (no modificada) |
| Tests (`tests/`) | ✅ 39/39 pasan |
| PDF de `document.md` | ⏳ Pendiente — exportar manualmente antes de entregar |
| Presentación | ⏳ Pendiente — crear antes de entregar |

---

## Cambios aplicados

### 1. `src/preprocessing.py` — Bug crítico corregido

**Cambio:** línea 363 — `FunctionTransformer(np.log1p)` →
`FunctionTransformer(np.log1p, feature_names_out="one-to-one")`.

**Por qué:** sin este parámetro, `get_feature_names_out()` lanzaba
`AttributeError: Estimator log does not provide get_feature_names_out`. Esto causaba:
- La pestaña "Variables más influyentes" de la app devolvía `None`.
- `SHAPExplainer.compute_factors()` producía factores vacíos.
- La pestaña "Explicación IA" generaba texto genérico sin factores SHAP reales.

**Impacto en métricas:** ninguno. Las métricas son idénticas a `reports/metrics_comparison.csv`
(la transformación `log1p` ya estaba presente; solo faltaba el parámetro de nombres).

Tras el fix se ejecutó `python -m src.train` para regenerar los archivos `.joblib`.

---

### 2. `tests/test_explainability.py` — Test alineado con contrato

**Cambio:** `test_returns_expected_structure_on_failure` (línea 161):
- Antes: `assert set(result.keys()) == {"positive", "negative"}`
- Después: `assert set(result.keys()) == {"positive", "negative", "error"}` +
  `assert result["error"] is not None`

**Por qué:** la implementación siempre devuelve la clave `"error"` (documentada en el
docstring de `compute_factors()`); el test estaba escrito antes de que se añadiera esa
clave al contrato.

**Resultado:** 39/39 tests pasan (antes: 37/39 pasaban; 2 fallaban por este motivo y por
el bug de `get_feature_names_out`).

---

### 3. `notebooks/01_housing_prices_proyecto_final.ipynb` — Mejoras visuales y narrativas

| Celda | ID | Cambio |
|---|---|---|
| Cell 0 — Portada | `b0d572f3` | Autoría correcta: 4 nombres oficiales |
| Cell 13 — EDA SalePrice | `cf1f7aa0` | Texto correcto con backticks restaurados; menciona log1p en producción |
| Cell 29 — Nota reproducibilidad | `nota-dup-src` | Backticks restaurados; aclara diferencias preprocesador notebook vs producción |
| Cell 47 — Residuos | `a94a1f78` | Backticks restaurados; corrige "mejora futura" → "ya implementado" |
| Cell 52 — Addendum rigor | `addendum-rigor-stat` | Elimina referencia a `VALIDACION_CIENTIFICA_FINAL.md` (archivo eliminado) |
| Cell 16 — Outliers | `318113af` | Reducido de 4 boxplots a 1 (GrLivArea) + scatter — elimina redundancia |
| Cell 41 — Antes/después tuning | `93dd44ac` | Eliminado subplot R² (información ya en tabla); conserva solo RMSE |
| Cell 50 — Importancia RF | `5de3e4f6` | Etiquetas en español legibles; elimina nombres internos `num__`/`cat__` |

El notebook fue re-ejecutado ("Run All") y confirmado determinista.

---

### 4. `docs/document.md` — Alineación con producción y autoría

| Sección | Cambio |
|---|---|
| Encabezado (l.5) | Autoría: 4 nombres oficiales |
| §2 tabla EDA | "mejora futura" → "implementado vía `TransformedTargetRegressor`" |
| §5.1 table RF | Métricas actualizadas a producción: MAE 17.212, RMSE 27.308, R² 0.9028 |
| §5.1 blockquote | Nota metodológica sobre configs B vs D/E — resuelve la auto-contradicción |
| §5.1.1 CV table RF | CV corregido: 30.007 ± 5.023, test 27.308 (config G producción) |
| §5.1.2 table | Nota metodológica + RF (raw) actualizado a 27.308 con etiquetas de config |
| §5.1.2 prosa | "0.4%" → "0.1%", valores 27.449 → 27.308 |
| §5.3 table RF | Test RMSE 27.449 → 27.308, gap 59.9% → 59.7% |
| §8 conclusiones | RF métricas: 27.449/0.902 → 27.308/0.903 |

---

### 5. `docs/DEFENSA/DEFENSA_RAPIDA.md` — Corrección de ablación y autoría

| Sección | Cambio |
|---|---|
| Encabezado (l.3) | Autoría: 4 nombres oficiales |
| §3 tabla ablación | **Reescrita completamente** con valores del CSV real (A=32 001, B=27 339, C=27 658, D/E=28 581, F=27 647, G=27 308); descripciones corregidas |
| §3 narrativa | Precisa impacto de cada config; documenta que D y E empeoran la LR |
| §7 tabla sobreajuste | RF test 27 449 → 27 308, gap 59.9% → 59.7% |
| §8 guion demo | Pestañas 3 y 4 marcadas como funcionales (no condicionales) |
| §9 respuesta métricas | Actualizada con referencias a configs A/B/D/E y CSV |
| §9 respuesta tests | "2 tests fallan" → "39/39 pasan; bug corregido" |

---

### 6. `README.md` — Autoría y estado de tests

| Elemento | Cambio |
|---|---|
| Encabezado | Añadido bloque de autoría con 4 nombres y afiliación |
| Sección de tests | "37 tests (8+29)" → "39 tests, todos pasan (10+29)" |

---

## Pendientes antes de la entrega

- [ ] **Exportar `docs/document.md` a PDF** (VS Code → Print → PDF, o Pandoc).
- [ ] **Crear presentación** con las diapositivas del proyecto.
- [ ] **Verificar Ollama activo** en el equipo de demo (`ollama list`).
- [ ] **Ensayar la demo** con los 3 escenarios del guion (compacta / familiar / lujo).

---

## Archivos NO modificados (por diseño)

- `reports/metrics_comparison.csv` — fuente de verdad de métricas
- `reports/ablation_study.csv` — resultados de ablación
- `src/train.py` — lógica de entrenamiento (excepto si se usa `--tune`)
- `app/` — toda la capa de aplicación Streamlit
- `data/` — datos crudos
- `requirements.txt` — dependencias
- Archivos de modelo (`models/*.joblib`) — regenerados automáticamente con el fix
