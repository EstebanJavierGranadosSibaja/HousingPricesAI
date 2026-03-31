# My IA

## Información

### Para el proyecto se debe usar PEAS

Framework peas:

- Performance
- environment
- actuators(output)
- sensors(input)

### Tipo de entorno

Observable - parcial Ajedrez - poker
Deterministico - estocástico Sudoku - mercado financiero
episódico - secuencial Clasificar email - conducir auto
estático - dinamico Crucigrama - videojuego en tiempo real
Discreto - continuo Juego damas - controlar brazo robótico

### mi tema

. Predicción de Precio de Viviendas
Tipo: Regresión
Descripción:
Predecir el precio de una vivienda según sus características.
Modelos:

Regresión Lineal
Random Forest

Resultado esperado:
El usuario ingresa:
Tamaño
Ubicación
Habitaciones

El sistema devuelve:
Precio estimado
Comparación entre modelos

### explicaciones

Claro, aquí tienes una explicación sencilla de ambos modelos:

**Regresión Lineal**  
La regresión lineal es un modelo estadístico que busca encontrar la relación entre una variable dependiente (lo que quieres predecir) y una o más variables independientes (las características o entradas). El modelo asume que esta relación es lineal, es decir, se puede representar con una recta.  
La fórmula básica es:  
$$
y = a + bx
$$  
donde $y$ es la variable a predecir, $x$ es la variable de entrada, $a$ es la intersección (bias) y $b$ es la pendiente (coeficiente). El modelo ajusta $a$ y $b$ para minimizar la diferencia entre los valores predichos y los reales.

**Random Forest**  
Random Forest es un modelo de aprendizaje automático basado en árboles de decisión. En vez de usar un solo árbol, construye muchos árboles (de ahí el "bosque") y cada uno hace una predicción. Luego, el modelo toma la predicción más común (para clasificación) o el promedio (para regresión) de todos los árboles.  
Las ventajas de Random Forest son que suele ser más preciso y robusto que un solo árbol, y es menos propenso a sobreajustar los datos.

¿Te gustaría ver ejemplos de código para alguno de estos modelos?

## Mi agente

### Cual es el agente?

El agente es se encarga de la prediccion del precio de una vivienda en base a sus características

### Cual es el entorno?

- Performance:
  - Presicion el la prediccion del precio
- Environment:
  - Datos de vivienda y sus características como referencias
- actuator:
  - Resultado precio estimado de la vivienda
  - Comparaciones entre otras viviendas de precio similar
- sensors:
  - entrada de textos con datos de vivienda: tamaño, año de construcción, ubicación y características de su entorno, habitaciones que tiene, estado actual...

### Cual es la funcion del agente?

Calcula en base a características ingresadas por el usuarios de la vivienda el precio estimado que tiene

### Cual es la medida de desempeño?

comparacion con datos de viviendas con características similares para determinar si el precio resultado es correcto. usando modelos de RMSE y MAE

"El RMSE penaliza más los errores grandes (sensible a valores atípicos) al elevar los residuos al cuadrado

MAE proporciona el promedio absoluto del error, siendo más interpretable y robusto ante valores atípicos."

<https://www.datacamp.com/es/tutorial/rmse>

<https://medium.com/@mondalsabbha/understanding-mae-mse-and-rmse-key-metrics-in-machine-learning-eeeff8bd1fac#:~:text=Objetivo%20del%20modelo&text=MAE%20:%20Ofrece%20una%20interpretaci%C3%B3n%20m%C3%A1s,penalizar%20m%C3%A1s%20los%20errores%20mayores>.

### Que tipo de entorno es ?

"El entorno es parcialmente observable, estocástico, secuencial, dinámico y continuo."

##

| Concepto    | En este caso                         |
| ----------- | ------------------------------------ |
| Agente      | Modelo de predicción                 |
| Entorno     | Mercado inmobiliario                 |
| Sensores    | Dataset / entrada de características |
| Actuadores  | Precio predicho                      |
| Performance | Error bajo (MAE, RMSE, etc.)         |

<https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data>

ejemplo_grid = [
    # 0  1  2  3  4  5  6  7  8  9
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9
]
