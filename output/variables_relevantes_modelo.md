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

- `OverallQual`: Calidad general
- `GrLivArea`: Área habitable sobre el nivel del suelo
- `TotalBsmtSF`: Área total de sótano
- `BsmtFinSF1`: Área terminada del sótano tipo 1
- `1stFlrSF`: Área primer piso
- `GarageCars`: Capacidad de autos en garaje
- `LotArea`: Área del lote
- `GarageArea`: Área del garaje
- `YearRemodAdd`: Año de remodelación
- `YearBuilt`: Año de construcción
- `FullBath`: Baños completos
- `OpenPorchSF`: Área de porche abierto
- `2ndFlrSF`: Área segundo piso
- `LotFrontage`: Metros lineales de frente de lote
- `GarageYrBlt`: Año de construcción del garaje
- `TotRmsAbvGrd`: Total de habitaciones sobre el nivel del suelo
- `BsmtUnfSF`: Área sin terminar del sótano
- `WoodDeckSF`: Área de terraza de madera
- `MoSold`: Mes de venta
- `OverallCond`: Condición general
- `SalePrice`: Precio de venta (variable objetivo)

## Exclusiones y filtros

- Se excluyen variables de identificación (`Id`), y aquellas con importancia predictiva muy baja según el análisis de Random Forest (menor o igual a 0.60%) como `MasVnrArea`, `BedroomAbvGr`, `Fireplaces`, `MSSubClass`, `YrSold`, `HalfBath`, y `BsmtFullBath`.
- Se excluyen variables con correlaciones negativas muy cercanas a cero con `SalePrice` como `Id` y `YrSold`.
- Se mantienen variables categóricas relevantes que, aunque no aparezcan en los análisis numéricos, aportan contexto valioso (ej. `Neighborhood`, `MSZoning`, `KitchenQual`, `SaleCondition`).
- Para variables categóricas, se recomienda agrupar valores poco frecuentes en "Other" o eliminarlos si son irrelevantes para evitar sobreajuste.
