# Caos determinista y pseudoaleatoriedad con el mapa logístico

## Ejecución

Solo se necesita tener Python 3 instalado. No hace falta usar la consola:

- **Windows:** doble clic en `main.py`, o clic derecho → *Editar con IDLE* → `F5`.
- **macOS / Linux:** abrir `main.py` con IDLE (u otro editor de Python) y ejecutarlo (`F5` en IDLE).

Si faltan `numpy`, `matplotlib` o `scipy`, el programa las instala automáticamente con `pip`
(requiere conexión a internet la primera vez). Al terminar, muestra un aviso y abre la carpeta
`salidas/`, que queda junto al programa, con las figuras y las tablas. Si ocurre un error, también se muestra en una ventana.

Desde la consola es equivalente:

```console
python main.py
```

La opción `--sin-ventana` omite el aviso final y no abre la carpeta. Todas las semillas son fijas,
así que cada ejecución reproduce exactamente los mismos resultados. Se verificó con Python 3.12.5
(numpy 1.26.4, matplotlib 3.8.3 y scipy 1.13.1) y con numpy 2.5, matplotlib 3.11 y scipy 1.18.
Los datos generados son idénticos; algunos estadísticos difieren recién en la cifra decimal 15 o 16.

## Estructura

| Archivo o carpeta | Función |
|---|---|
| `main.py` | Programa principal. Recurrencia, cálculo de los experimentos A–H, pruebas estadísticas y exportación de CSV. |
| `configuracion.py` | Semillas, valores de `r`, tamaños de las mallas y de la muestra, transitorios, tolerancias y resolución de figuras. |
| `graficos.py` | Figuras de Matplotlib. |
| `dependencias.py` | Instala las bibliotecas que falten y muestra el aviso final. |
| `salidas/figuras/` | Quince figuras PNG de 300 ppp. |
| `salidas/tablas/` | Veintiún CSV en UTF-8, separados por comas y con punto decimal. |

## Experimentos

| Experimento | Contenido |
|---|---|
| A. Trayectorias | Una figura con las primeras 120 iteraciones para ocho valores de `r`. Los CSV incluyen las órbitas completas y el diagnóstico tardío. |
| B. Solución explícita | Comparación con la recurrencia para `r = 4`, con una referencia decimal de 100 dígitos. Se grafica el acuerdo temprano y la pérdida de precisión tardía. |
| C. Bifurcación | Malla general de 6001 valores entre 2.5 y 4 y malla ampliada alrededor de la ventana de período 3. Cada CSV tiene una fila por `r` y una columna `x_n` por iteración retenida. |
| D. Ventana de período 3 | Localización del tramo con período 3 detectado, contraste con un caso caótico cercano y controles con distintos transitorios en los bordes. |
| E. Condiciones iniciales | Cuatro figuras por región. Los CSV reúnen trayectorias iniciales y tardías, diagnósticos por semilla y comparaciones por pares con alineación de ciclos. |
| F. Sensibilidad (parte 2) | Semilla `0.314159265` y catorce semillas que suben en una unidad la cifra decimal `k = 2, …, 15`. Trayectorias de 100 iteraciones, distancia en escala logarítmica, primera iteración con distancia mayor que 0.1 y ajuste lineal frente a la predicción `log2(0.1/δ)`. |
| G. Distribución y dependencia (parte 3) | `x_1000` para 10 000 semillas de la malla determinista `(i + 1/2)/20000`. Histograma, función de distribución empírica, media, varianza, cuantiles, KS frente a Uniforme(0, 1) y Beta(1/2, 1/2). Autocorrelación, Ljung–Box, rachas y χ² de independencia por pares, tanto para la muestra como para una trayectoria única. |
| H. Uniformización (parte 4) | `u = (2/π)·arcsin(√x)` sobre la muestra de G: histograma, función de distribución, estadísticos, KS frente a Uniforme(0, 1) y diagrama `u_n` frente a `u_{n+1}` (mapa tienda). |

