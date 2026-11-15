# Caos determinista y pseudoaleatoriedad con el mapa logístico

## Instalación y ejecución

Se verificó con Python 3.11.5. Desde este directorio, instalá las dependencias y ejecutá el programa principal.

```console
python -m pip install -r requirements.txt
python main.py
```

`python main.py` regenera las figuras y las tablas. Se guardan junto al programa, en `salidas/`,
con semillas deterministas. El comando ejecuta todos los experimentos, de A a I. Los resultados
de sensibilidad, distribución, dependencia y uniformización se guardan con los prefijos `f_`,
`g_`, `h_` e `i_`, respectivamente.

## Estructura

| Archivo o carpeta | Función |
|---|---|
| `main.py` | Recurrencia, ejecución de los experimentos A–I y exportación de CSV. |
| `configuracion.py` | Semillas, valores de `r`, tamaños de las mallas, transitorios, tolerancias y resolución de figuras. |
| `graficos.py` | Figuras de Matplotlib. |
| `estadistica.py` | Estadísticos descriptivos y contrastes de Kolmogórov–Smirnov y Ljung–Box con SciPy. |
| `requirements.txt` | Versiones de NumPy, Matplotlib y SciPy usadas por el programa. |
| `salidas/figuras/` | 16 figuras PNG de 300 ppp con la configuración por defecto. |
| `salidas/tablas/` | 21 CSV en UTF-8, separados por comas y con punto decimal. |

## Experimentos

| Experimento | Contenido |
|---|---|
| A. Trayectorias | Una figura con las primeras 120 iteraciones para ocho valores de `r`. Los CSV incluyen las órbitas completas y el diagnóstico tardío. |
| B. Solución explícita | Comparación con la recurrencia para `r = 4`, con una referencia decimal de 100 dígitos. Se grafica el acuerdo temprano y la pérdida de precisión tardía. |
| C. Bifurcación | Malla general de 6001 valores entre 2.5 y 4 y malla ampliada alrededor de la ventana de período 3. Cada CSV tiene una fila por `r` y una columna `x_n` por iteración retenida. |
| D. Ventana de período 3 | Localización del tramo con período 3 detectado, contraste con un caso caótico cercano y controles con distintos transitorios en los bordes. |
| E. Condiciones iniciales | Cuatro figuras por región. Los CSV reúnen trayectorias iniciales y tardías, diagnósticos por semilla y comparaciones por pares con alineación de ciclos. |
| F. Sensibilidad (parte 2 de la letra) | Compara nueve perturbaciones de una semilla para `r = 4` durante 100 iteraciones y registra cuándo cada distancia supera `0.1`. |
| G. Distribución (partes 3.1 y 3.2) | Toma `x_1000` de 10 000 semillas deterministas y compara la muestra con la uniforme y la Beta(1/2, 1/2) mediante gráficos, estadísticos descriptivos y el contraste KS. |
| H. Dependencia (parte 3.3) | Compara la muestra ordenada por semilla y una trayectoria única mediante autocorrelaciones y el contraste de Ljung–Box con 20 rezagos. |
| I. Uniformización (parte 4) | Transforma la misma muestra de G y compara su distribución con la uniforme mediante un histograma, estadísticos descriptivos y el contraste KS. También calcula Ljung–Box. |

Ljung–Box evalúa la autocorrelación en los rezagos elegidos. Los valores p del KS se calculan
con la calibración habitual del contraste y se interpretan teniendo en cuenta que la muestra
se generó con una malla determinista de semillas.

