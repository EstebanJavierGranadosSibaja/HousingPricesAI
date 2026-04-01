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

- `MSSubClass`: Tipo de vivienda (excluyendo valores poco frecuentes o irrelevantes)
- `MSZoning`: Clasificación de zona (mantener solo RL, RM, FV, RH)
- `LotFrontage`: Metros lineales de frente de lote
- `LotArea`: Área del lote
- `Neighborhood`: Vecindario
- `OverallQual`: Calidad general
- `OverallCond`: Condición general
- `YearBuilt`: Año de construcción
- `YearRemodAdd`: Año de remodelación
- `TotalBsmtSF`: Área total de sótano
- `1stFlrSF`: Área primer piso
- `2ndFlrSF`: Área segundo piso
- `GrLivArea`: Área habitable sobre el nivel del suelo
- `FullBath`: Baños completos
- `HalfBath`: Medios baños
- `BedroomAbvGr`: Dormitorios sobre el nivel del suelo
- `KitchenAbvGr`: Cocinas sobre el nivel del suelo
- `KitchenQual`: Calidad de la cocina
- `TotRmsAbvGrd`: Total de habitaciones sobre el nivel del suelo
- `Fireplaces`: Número de chimeneas
- `GarageCars`: Capacidad de autos en garaje
- `GarageArea`: Área del garaje
- `SaleType`: Tipo de venta (mantener solo WD, New, COD, Con, CWD)
- `SaleCondition`: Condición de venta (mantener solo Normal, Partial)
- `SalePrice`: Precio de venta (variable objetivo)

## Exclusiones y filtros

- Se excluyen variables de identificación (Id), variables con muchos valores faltantes o poco informativas (Alley, PoolQC, Fence, MiscFeature).
- Se excluyen variables con alta redundancia o difícil interpretación directa para el modelo base.
- Para variables categóricas, se recomienda agrupar valores poco frecuentes en "Other" o eliminarlos si son irrelevantes.
