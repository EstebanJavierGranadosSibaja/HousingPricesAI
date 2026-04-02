# Variables relevantes para el modelo de predicción de precios de viviendas

## Justificación de la selección de variables

La selección de variables para el modelo se basa en los siguientes criterios:

1. **Importancia según Random Forest**: Se analizaron los resultados de importancia de variables generados por un modelo de Random Forest (`importancia_rf.csv`). Variables como `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `GarageArea`, `1stFlrSF`, `YearBuilt`, y `YearRemodAdd` destacan por su alta contribución a la predicción del precio.

2. **Correlación con el precio de venta**: Se consideró la correlación lineal de cada variable con el precio de venta (`correlacion_SalePrice.csv`). Variables como `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `GarageArea`, y `1stFlrSF` presentan correlaciones superiores al 60%, lo que indica una fuerte relación con el objetivo.

3. **Significado y utilidad práctica**: Se revisó el diccionario de variables refinadas y el reporte de variables para asegurar que las variables seleccionadas sean interpretables, relevantes y disponibles en los datos. Se priorizaron variables que reflejan características estructurales, de calidad y ubicación de la vivienda.

4. **Variables categóricas relevantes**: Se incluyeron variables categóricas con suficiente frecuencia y relevancia, como `Neighborhood`, `MSZoning`, `KitchenQual`, y `SaleCondition`, ya que aportan información contextual y de calidad que no está reflejada en las variables numéricas.

5. **Exclusión de variables**: Se descartaron variables con muchos valores nulos, baja frecuencia, redundancia, difícil interpretación o que no aportan valor predictivo (por ejemplo, `Id`, `Alley`, `PoolQC`, `Fence`, `MiscFeature`).

6. **Filtrado de valores**: Para variables categóricas, se recomienda agrupar valores poco frecuentes en "Other" o eliminarlos si son irrelevantes, para evitar sobreajuste y mejorar la robustez del modelo.

Esta selección busca un equilibrio entre capacidad predictiva, interpretabilidad y facilidad de preprocesamiento, facilitando la construcción de modelos robustos y explicables.

Este archivo contiene la selección de variables y valores recomendados para usar en un modelo de regresión lineal o Random Forest, filtrando tanto variables como valores poco informativos o irrelevantes.

## Variables seleccionadas

**Variables Numéricas (12):**

- `OverallQual`: Calidad general de materiales y acabados (#1 en importancia)
- `GrLivArea`: Área habitable total sobre el nivel del suelo
- `TotalBsmtSF`: Área total del sótano
- `1stFlrSF`: Área del primer piso
- `GarageCars`: Capacidad de autos en el garaje (más correlacionado que GarageArea)
- `LotArea`: Tamaño del lote
- `YearBuilt`: Año de construcción (indica modernidad)
- `YearRemodAdd`: Año de remodelación
- `FullBath`: Número de baños completos
- `TotRmsAbvGrd`: Total de habitaciones (sin incluir baños)
- `Fireplaces`: Número de chimeneas (indicador de lujo)
- `OverallCond`: Condición general de la vivienda

**Variables Categóricas (8):**

- `Neighborhood`: Ubicación física (crítico para el valor del suelo)
- `ExterQual`: Calidad de los materiales exteriores
- `KitchenQual`: Calidad de la cocina
- `BsmtQual`: Calidad/altura del sótano
- `Foundation`: Tipo de cimentación (indicador estructural)
- `MSZoning`: Clasificación de zona (Residencial, Comercial, etc.)
- `SaleCondition`: Condición de la venta
- `CentralAir`: Presencia de aire acondicionado central (binario)

## Exclusiones y Justificación de Refinamiento (Simplificación del Modelo)

Para mejorar la robustez y evitar la multicolinealidad, se han realizado los siguientes ajustes:

1. **Eliminación por Redundancia**: Se eliminó `GarageYrBlt` (redundante con `YearBuilt`) y `GarageArea` (redundante con `GarageCars`).
2. **Eliminación por Bajo Impacto/Ruido**: Se eliminaron `MoSold` (mes de venta) y `YrSold` debido a su bajísima correlación con el precio final.
3. **Eliminación por Variables de Desglose**: Se prefiere usar `TotalBsmtSF` en lugar de sus desgloses (`BsmtFinSF1`, `BsmtUnfSF`) para evitar ruido estadístico, manteniendo `BsmtQual` para capturar la "calidad" de ese espacio.
4. **Inclusión de Variables de Valor Real**: Se añadieron `CentralAir` y `Fireplaces` porque, aunque no siempre dominan en importancia estadística lineal, son características de alto valor para los compradores.
5. **Simplificación de Áreas Exteriores**: Se eliminaron indicadores de porche específicos (`OpenPorchSF`, `WoodDeckSF`) para centrar el modelo en la estructura principal.
