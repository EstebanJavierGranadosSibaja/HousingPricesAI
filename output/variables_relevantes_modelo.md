# Variables relevantes para el modelo de predicción de precios de viviendas

Este archivo contiene la selección de variables y valores recomendados para usar en un modelo de regresión lineal o Random Forest, filtrando tanto variables como valores poco informativos o irrelevantes.

## Variables seleccionadas

- `MSSubClass`:Tipo de vivienda (excluyendo valores poco frecuentes o irrelevantes)
- `MSZoning`:Clasificación de zona (mantener solo RL, RM, FV, RH)
- `LotFrontage`:Metros lineales de frente de lote
- `LotArea`:Área del lote
- `Neighborhood`:Vecindario
- `OverallQual`:Calidad general
- `OverallCond`:Condición general
- `YearBuilt`:Año de construcción
- `YearRemodAdd`:Año de remodelación
- `TotalBsmtSF`:Área total de sótano
- `1stFlrSF`:Área primer piso
- `2ndFlrSF`:Área segundo piso
- `GrLivArea`:Área habitable sobre el nivel del suelo
- `FullBath`:Baños completos
- `HalfBath`:Medios baños
- `BedroomAbvGr`:Dormitorios sobre el nivel del suelo
- `KitchenAbvGr`:Cocinas sobre el nivel del suelo
- `KitchenQual`:Calidad de la cocina
- `TotRmsAbvGrd`:Total de habitaciones sobre el nivel del suelo
- `Fireplaces`:Número de chimeneas
- `GarageCars`:Capacidad de autos en garaje
- `GarageArea`:Área del garaje
- `SaleType`:Tipo de venta (mantener solo WD, New, COD, Con, CWD)
- `SaleCondition`:Condición de venta (mantener solo Normal, Partial)
- `SalePrice`:Precio de venta (variable objetivo)

## Exclusiones y filtros

- Se excluyen variables de identificación (Id), variables con muchos valores faltantes o poco informativas (Alley, PoolQC, Fence, MiscFeature).
- Se excluyen variables con alta redundancia o difícil interpretación directa para el modelo base.
- Para variables categóricas, se recomienda agrupar valores poco frecuentes en "Other" o eliminarlos si son irrelevantes.

## Ejemplo de variables y valores válidos

| Variable      | Valores válidos principales               |
| ------------- | ----------------------------------------- |
| MSZoning      | RL, RM, FV, RH                            |
| SaleType      | WD, New, COD, Con, CWD                    |
| SaleCondition | Normal, Partial                           |
| KitchenQual   | Ex, Gd, TA, Fa                            |
| Neighborhood  | (todos, pero se pueden agrupar los raros) |

## Notas

- Antes de entrenar el modelo, realiza limpieza de datos y codificación de variables categóricas.
- Puedes ajustar la lista según el análisis exploratorio y la importancia de variables en tu modelo.
