# Agente de prediccion de precios para viviendas (randon forest)

**Planteamiento del Problema**

Tenemos la necesidad de saber una valoracion de cuanto cuesta una casa segun sus características, se compara como referencia otras casas para terminar el precio.

Se busca resolver el problema de incertidumbre ante alguna compra o venta de propiedad para saber su precio estimado.

**Indice**

- [Agente de prediccion de precios para viviendas (randon forest)](#agente-de-prediccion-de-precios-para-viviendas-randon-forest)
  - [Marco conceptual](#marco-conceptual)
  - [librerías](#librerías)
  - [Ajustes de filas y columnas](#ajustes-de-filas-y-columnas)
    - [Filtrado de datos](#filtrado-de-datos)
  - [Cargar el dataset](#cargar-el-dataset)
    - [Análisis Exploratorio](#análisis-exploratorio)
  - [Selección y Entrenamiento del Modelo](#selección-y-entrenamiento-del-modelo)
  - [Evaluación del Modelo](#evaluación-del-modelo)
  - [testear el modelo creado](#testear-el-modelo-creado)

## Marco conceptual

- Agrupamiento de datos raros: busca agrupar variables raras poco frecuentes
- Varianza: define que tan diversos son los datos
  - Alta: significa que hay diversidad (son datos que aportan)
  - Baja: significa que no hay diversidad (no aporta nada)
- Percentil: divide los datos en porcentajes, controla valores extremos
  - por debajo de x% se encuentra los datos
  - entre x1% y x2% se encuentra los datos

## librerías

## Ajustes de filas y columnas

### Filtrado de datos

| Paso | Variables | Qué hace | Por qué es correcto |
|---|---|---|---|
| `agrupar_categorias_raras` | Categóricas | Reemplaza categorías <1% por `"Otro"` | No pierde filas; aprende umbrales solo en train |
| `eliminar_baja_varianza` | Todas | Elimina columnas donde >85% es un mismo valor | Una columna casi constante no aporta información predictiva |
| `winsorizar_numericas` | Numéricas | Recorta valores extremos al percentil 1%–99% | Reduce el efecto de outliers sin eliminar observaciones |

se busca reducir el ruido de la informacion para determinar variables importantes

## Cargar el dataset

### Análisis Exploratorio

## Selección y Entrenamiento del Modelo

Explica por qué elegiste Random Forest, cómo configuraste el modelo y cómo lo entrenaste.

## Evaluación del Modelo

Presenta los resultados de las pruebas (métricas como MAE, RMSE, etc.) y qué significan.

## testear el modelo creado
