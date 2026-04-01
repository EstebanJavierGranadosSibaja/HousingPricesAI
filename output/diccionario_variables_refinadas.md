# Diccionario de variables refinadas

A continuación se describen las variables seleccionadas tras el preprocesamiento

- con su significado
- traducción y posibles valores/opciones.

## MSSubClass

**Significado:** Identifica el tipo de vivienda involucrada en la venta.
**Traducción:** Tipo de vivienda.
**Opciones:**

- 20: 1 piso
- 1946 y más reciente
- 30: 1 piso
- 1945 y más antiguo
- 40: 1 piso con ático terminado
- 45: 1 y medio piso sin terminar
- 50: 1 y medio piso terminado
- 60: 2 pisos
- 1946 y más reciente
- 70: 2 pisos
- 1945 y más antiguo
- 75: 2 y medio pisos
- 80: Dividido o multinivel
- 85: Vestíbulo dividido
- 90: Dúplex
- todos los estilos y edades
- 120: 1 piso PUD
- 1946 y más reciente
- 150: 1 y medio piso PUD
- 160: 2 pisos PUD
- 1946 y más reciente
- 180: PUD multinivel
- 190: Conversión a 2 familias
- todos los estilos y edades

## MSZoning

**Significado:** Clasificación de zonificación general de la venta.
**Traducción:** Zonificación.
**Opciones:**

- RL (Residencial baja densidad)
- RM (Residencial densidad media)
- C (Comercial)
- FV (Residencial de villa flotante)
- RH (Residencial alta densidad)

## LotFrontage

**Significado:** Metros lineales de calle conectados a la propiedad.
**Traducción:** Frente de lote (pies).
**Opciones:**

- Numérico.

## LotArea

**Significado:** Tamaño del lote en pies cuadrados.
**Traducción:** Área del lote.
**Opciones:**

- Numérico.

## Street

**Significado:** Tipo de acceso vial a la propiedad.
**Traducción:** Calle.
**Opciones:**

- Pave: Pavimentado
- Grvl: Grava

## Alley

**Significado:** Tipo de acceso por callejón a la propiedad.
**Traducción:** Callejón.
**Opciones:**

- Grvl: Grava
- Pave: Pavimentado

## LotShape

**Significado:** Forma general de la propiedad.
**Traducción:** Forma del lote.
**Opciones:**

- Reg (Regular)
- IR1 (Ligeramente irregular)
- IR2 (Moderadamente irregular)
- IR3 (Irregular)

## LandContour

**Significado:** Nivel de planitud de la propiedad.
**Traducción:** Contorno del terreno.
**Opciones:**

- Lvl (Casi plano)
- Bnk (Elevado)
- Low (Depresión)
- HLS (Ladera)

## Utilities

**Significado:** Tipo de servicios disponibles.
**Traducción:** Servicios públicos.
**Opciones:**

- AllPub (Todos los servicios públicos)
- NoSeWa (Solo electricidad y gas)

## LotConfig

**Significado:** Configuración del lote.
**Traducción:** Configuración del lote.
**Opciones:**

- Inside (Lote interior)
- FR2 (Frente en 2 lados)
- Corner (Lote esquina)
- CulDSac (Cul-de-sac)
- FR3 (Frente en 3 lados)

## LandSlope

**Significado:** Pendiente de la propiedad.
**Traducción:** Pendiente del terreno.
**Opciones:**

- Gtl (Pendiente suave)
- Mod (Pendiente moderada)
- Sev (Pendiente severa)

## Neighborhood

**Significado:** Ubicación física dentro de los límites de Ames.
**Traducción:** Vecindario.
**Opciones:**

- CollgCr
- Veenker
- Crawfor
- NoRidge
- Mitchel
- Somerst
- NWAmes
- OldTown
- BrkSide
- Sawyer
- NridgHt
- NAmes
- SawyerW
- IDOTRR
- MeadowV
- Edwards
- Timber
- Gilbert
- StoneBr
- ClearCr
- NPkVill
- Blmngtn
- BrDale
- SWISU
- Blueste

## Condition1

**Significado:** Proximidad a diversas condiciones.
**Traducción:** Condición 1.
**Opciones:**

- Norm
- Feedr
- PosN
- Artery
- RRAe
- RRNn
- RRAn
- PosA
- RRNe

## Condition2

**Significado:** Proximidad a diversas condiciones (si hay más de una presente).
**Traducción:** Condición 2.
**Opciones:**

- Norm
- Artery
- RRNn
- Feedr
- PosN
- PosA
- RRAn
- RRAe

## BldgType

**Significado:** Tipo de vivienda.
**Traducción:** Tipo de edificio.
**Opciones:**

- 1Fam (Casa unifamiliar)
- 2fmCon (Conversión a dos familias)
- Duplex (Dúplex)
- TwnhsE (Casa adosada final)
- Twnhs (Casa adosada interior)

## HouseStyle

**Significado:** Estilo de vivienda.
**Traducción:** Estilo de casa.
**Opciones:**

- 2Story
- 1Story
- 1.5Fin
- 1.5Unf
- SFoyer
- SLvl
- 2.5Unf
- 2.5Fin

## OverallQual

**Significado:** Califica el material y acabado general de la casa.
**Traducción:** Calidad general.
**Opciones:**

- 1 (Muy pobre) a 10 (Muy excelente)

## OverallCond

**Significado:** Califica la condición general de la casa.
**Traducción:** Condición general.
**Opciones:**

- 1 (Muy pobre) a 10 (Muy excelente)

## YearBuilt

**Significado:** Fecha de construcción original.
**Traducción:** Año de construcción.
**Opciones:**

- Numérico.

## YearRemodAdd

**Significado:** Fecha de remodelación (igual a la de construcción si no hubo remodelaciones).
**Traducción:** Año de remodelación.
**Opciones:**

- Numérico.

## RoofStyle

**Significado:** Tipo de techo.
**Traducción:** Estilo de techo.
**Opciones:**

- Gable (A dos aguas)
- Hip (A cuatro aguas)
- Gambrel (Granero)
- Mansard (Mansarda)
- Flat (Plano)
- Shed (Cobertizo)

## RoofMatl

**Significado:** Material del techo.
**Traducción:** Material del techo.
**Opciones:**

- CompShg
- WdShngl
- Metal
- WdShake
- Membran
- Tar&Grv
- Roll
- ClyTile

## Exterior1st

**Significado:** Cubierta exterior de la casa.
**Traducción:** Exterior 1.
**Opciones:**

- VinylSd
- MetalSd
- Wd Sdng
- HdBoard
- BrkFace
- WdShing
- CemntBd
- Plywood
- AsbShng
- Stucco
- BrkComm
- AsphShn
- Stone
- ImStucc
- CBlock

## Exterior2nd

**Significado:** Cubierta exterior de la casa (si hay más de un material).
**Traducción:** Exterior 2.
**Opciones:**

- VinylSd
- MetalSd
- Wd Shng
- HdBoard
- Plywood
- Wd Sdng
- CmentBd
- BrkFace
- Stucco
- AsbShng
- Brk Cmn
- ImStucc
- AsphShn
- Stone
- Other
- CBlock

## MasVnrType

**Significado:** Tipo de revestimiento de mampostería.
**Traducción:** Tipo de revestimiento.
**Opciones:**

- BrkFace
- Stone
- BrkCmn

## MasVnrArea

**Significado:** Área de revestimiento de mampostería en pies cuadrados.
**Traducción:** Área de revestimiento.
**Opciones:**

- Numérico.

## ExterQual

**Significado:** Evalúa la calidad del material en el exterior.
**Traducción:** Calidad exterior.
**Opciones:**

- Ex (Excelente)
- Gd (Bueno)
- TA (Promedio)
- Fa (Regular)

## ExterCond

**Significado:** Evalúa la condición actual del material en el exterior.
**Traducción:** Condición exterior.
**Opciones:**

- TA (Promedio)
- Gd (Bueno)
- Fa (Regular)
- Po (Pobre)
- Ex (Excelente)

## Foundation

**Significado:** Tipo de cimiento.
**Traducción:** Fundación.
**Opciones:**

- PConc
- CBlock
- BrkTil
- Wood
- Slab
- Stone

## BsmtQual

**Significado:** Evalúa la altura del sótano.
**Traducción:** Calidad del sótano.
**Opciones:**

- Gd (Bueno)
- TA (Típico)
- Ex (Excelente)
- Fa (Regular)

## BsmtCond

**Significado:** Evalúa la condición general del sótano.
**Traducción:** Condición del sótano.
**Opciones:**

- TA (Típico)
- Gd (Bueno)
- Fa (Regular)
- Po (Pobre)

## BsmtExposure

**Significado:** Se refiere a paredes de salida o nivel jardín.
**Traducción:** Exposición del sótano.
**Opciones:**

- No
- Gd
- Mn
- Av

## BsmtFinType1

**Significado:** Calificación del área terminada del sótano.
**Traducción:** Tipo de acabado sótano 1.
**Opciones:**

- GLQ
- ALQ
- Unf
- Rec
- BLQ
- LwQ

## BsmtFinSF1

**Significado:** Tipo 1 de pies cuadrados terminados.
**Traducción:** Área terminada sótano 1.
**Opciones:**

- Numérico.

## BsmtFinType2

**Significado:** Calificación del área terminada del sótano, si hay varios tipos.
**Traducción:** Tipo de acabado sótano 2.
**Opciones:**

- Unf
- BLQ
- ALQ
- Rec
- LwQ
- GLQ

## BsmtFinSF2

**Significado:** Tipo 2 de pies cuadrados terminados.
**Traducción:** Área terminada sótano 2.
**Opciones:**

- Numérico.

## BsmtUnfSF

**Significado:** Pies cuadrados sin terminar del sótano.
**Traducción:** Área sin terminar sótano.
**Opciones:**

- Numérico.

## TotalBsmtSF

**Significado:** Total de pies cuadrados del sótano.
**Traducción:** Área total sótano.
**Opciones:**

- Numérico.

## Heating

**Significado:** Tipo de calefacción.
**Traducción:** Calefacción.
**Opciones:**

- GasA
- GasW
- Grav
- Wall
- OthW
- Floor

## HeatingQC

**Significado:** Calidad y condición de la calefacción.
**Traducción:** Calidad calefacción.
**Opciones:**

- Ex
- Gd
- TA
- Fa
- Po

## CentralAir

**Significado:** Aire acondicionado central.
**Traducción:** Aire acondicionado central.
**Opciones:**

- Y (Sí)
- N (No)

## Electrical

**Significado:** Sistema eléctrico.
**Traducción:** Eléctrico.
**Opciones:**

- SBrkr
- FuseF
- FuseA
- FuseP
- Mix

## 1stFlrSF

**Significado:** Pies cuadrados del primer piso.
**Traducción:** Área primer piso.
**Opciones:**

- Numérico.

## 2ndFlrSF

**Significado:** Pies cuadrados del segundo piso.
**Traducción:** Área segundo piso.
**Opciones:**

- Numérico.

## LowQualFinSF

**Significado:** Pies cuadrados terminados de baja calidad, todos los pisos.
**Traducción:** Área baja calidad.
**Opciones:**

- Numérico.

## GrLivArea

**Significado:** Área habitable sobre el nivel del suelo en pies cuadrados.
**Traducción:** Área habitable.
**Opciones:**

- Numérico.

## BsmtFullBath

**Significado:** Baños completos en el sótano.
**Traducción:** Baños completos sótano.
**Opciones:**

- 1.0
- 0.0
- 2.0
- 3.0

## BsmtHalfBath

**Significado:** Medios baños en el sótano.
**Traducción:** Medios baños sótano.
**Opciones:**

- 0.0
- 1.0
- 2.0

## FullBath

**Significado:** Baños completos sobre el nivel del suelo.
**Traducción:** Baños completos.
**Opciones:**

- 2
- 1
- 3
- 0
- 4

## HalfBath

**Significado:** Medios baños sobre el nivel del suelo.
**Traducción:** Medios baños.
**Opciones:**

- 1
- 0
- 2

## BedroomAbvGr

**Significado:** Habitaciones sobre el nivel del suelo

- no incluye sótano.
**Traducción:** Dormitorios.
**Opciones:**

- 3
- 4
- 1
- 2
- 0
- 5
- 6
- 8

## KitchenAbvGr

**Significado:** Cocinas sobre el nivel del suelo.
**Traducción:** Cocinas.
**Opciones:**

- 1
- 2
- 3
- 0

## KitchenQual

**Significado:** Calidad de la cocina.
**Traducción:** Calidad cocina.
**Opciones:**

- Gd (Bueno)
- TA (Promedio)
- Ex (Excelente)
- Fa (Regular)

## TotRmsAbvGrd

**Significado:** Total de habitaciones sobre el nivel del suelo

- no incluye baños.
**Traducción:** Total habitaciones.
**Opciones:**

- 2
- 3
- 4
- 8
- 5
- 6
- 7
- 9
- 10
- 11
- 12
- 13
- 14
- 15

## Functional

**Significado:** Funcionalidad del hogar, se asume típica salvo deducciones.
**Traducción:** Funcionalidad.
**Opciones:**

- Typ
- Min1
- Maj1
- Min2
- Mod
- Maj2
- Sev

## Fireplaces

**Significado:** Número de chimeneas.
**Traducción:** Chimeneas.
**Opciones:**

- 0
- 1
- 2
- 3
- 4

## FireplaceQu

**Significado:** Calidad de la chimenea.
**Traducción:** Calidad chimenea.
**Opciones:**

- TA
- Gd
- Fa
- Ex
- Po

## GarageType

**Significado:** Ubicación del garaje.
**Traducción:** Tipo de garaje.
**Opciones:**

- Attchd
- Detchd
- BuiltIn
- CarPort
- Basment
- 2Types

## GarageYrBlt

**Significado:** Año de construcción del garaje.
**Traducción:** Año garaje.
**Opciones:**

- Numérico.

## GarageFinish

**Significado:** Acabado interior del garaje.
**Traducción:** Acabado garaje.
**Opciones:**

- RFn
- Unf
- Fin

## GarageCars

**Significado:** Capacidad del garaje en número de autos.
**Traducción:** Autos garaje.
**Opciones:**

- 2.0
- 3.0
- 1.0
- 0.0
- 4.0
- 5.0

## GarageArea

**Significado:** Tamaño del garaje en pies cuadrados.
**Traducción:** Área garaje.
**Opciones:**

- Numérico.

## GarageQual

**Significado:** Calidad del garaje.
**Traducción:** Calidad garaje.
**Opciones:**

- TA
- Fa
- Gd
- Ex
- Po

## GarageCond

**Significado:** Condición del garaje.
**Traducción:** Condición garaje.
**Opciones:**

- TA
- Fa
- Gd
- Po
- Ex

## PavedDrive

**Significado:** Entrada pavimentada.
**Traducción:** Entrada pavimentada.
**Opciones:**

- Y (Pavimentada)
- N (Tierra/Grava)
- P (Pavimentación parcial)

## WoodDeckSF

**Significado:** Área de terraza de madera en pies cuadrados.
**Traducción:** Terraza madera.
**Opciones:**

- Numérico.

## OpenPorchSF

**Significado:** Área de porche abierto en pies cuadrados.
**Traducción:** Porche abierto.
**Opciones:**

- Numérico.

## EnclosedPorch

**Significado:** Área de porche cerrado en pies cuadrados.
**Traducción:** Porche cerrado.
**Opciones:**

- Numérico.

## 3SsnPorch

**Significado:** Área de porche de tres estaciones en pies cuadrados.
**Traducción:** Porche 3 estaciones.
**Opciones:**

- Numérico.

## ScreenPorch

**Significado:** Área de porche con malla en pies cuadrados.
**Traducción:** Porche con malla.
**Opciones:**

- Numérico.

## PoolArea

**Significado:** Área de piscina en pies cuadrados.
**Traducción:** Área piscina.
**Opciones:**

- Numérico.

## PoolQC

**Significado:** Calidad de la piscina.
**Traducción:** Calidad piscina.
**Opciones:**

- Ex
- Fa
- Gd

## Fence

**Significado:** Calidad de la cerca.
**Traducción:** Calidad cerca.
**Opciones:**

- MnPrv
- GdWo
- GdPrv
- MnWw

## MiscFeature

**Significado:** Característica miscelánea no cubierta en otras categorías.
**Traducción:** Característica miscelánea.
**Opciones:**

- Shed
- Gar2
- Othr
- TenC

## MiscVal

**Significado:** Valor en dólares de la característica miscelánea.
**Traducción:** Valor misceláneo.
**Opciones:**

- Numérico.

## MoSold

**Significado:** Mes de venta.
**Traducción:** Mes vendido.
**Opciones:**

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12

## YrSold

**Significado:** Año de venta.
**Traducción:** Año vendido.
**Opciones:**

- 2008
- 2007
- 2006
- 2009
- 2010

## SaleType

**Significado:** Tipo de venta.
**Traducción:** Tipo de venta.
**Opciones:**

- WD
- New
- COD
- ConLD
- ConLI
- CWD
- ConLw
- Con
- Oth

## SaleCondition

**Significado:** Condición de la venta.
**Traducción:** Condición venta.
**Opciones:**

- Normal
- Abnorml
- Partial
- AdjLand
- Alloca
- Family

## SalePrice

**Significado:** Precio de venta de la vivienda.
**Traducción:** Precio de venta.
**Opciones:**

- Numérico.
