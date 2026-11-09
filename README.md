# Caos determinista y pseudoaleatoriedad con el mapa logístico

## Instalación y ejecución

Se verificó con Python 3.11.5. Desde este directorio, instalá las dependencias y ejecutá el programa principal.

```console
python -m pip install -r requirements.txt
python main.py
```

`python main.py` regenera las figuras y las tablas. Se guardan junto al programa, en `salidas/`,
con semillas fijas.

## Estructura

| Archivo o carpeta | Función |
|---|---|
| `main.py` | Recurrencia, cálculo de los experimentos A–E y exportación de CSV. |
| `configuracion.py` | Semillas, valores de `r`, tamaños de las mallas, transitorios, tolerancias y resolución de figuras. |
| `graficos.py` | Figuras de Matplotlib. |
| `salidas/figuras/` | Nueve figuras PNG de 300 ppp. |
| `salidas/tablas/` | Doce CSV en UTF-8, separados por comas y con punto decimal. |

## Experimentos

| Experimento | Contenido |
|---|---|
| A. Trayectorias | Una figura con las primeras 120 iteraciones para ocho valores de `r`. Los CSV incluyen las órbitas completas y el diagnóstico tardío. |
| B. Solución explícita | Comparación con la recurrencia para `r = 4`, con una referencia decimal de 100 dígitos. Se grafica el acuerdo temprano y la pérdida de precisión tardía. |
| C. Bifurcación | Malla general de 6001 valores entre 2.5 y 4 y malla ampliada alrededor de la ventana de período 3. Cada CSV tiene una fila por `r` y una columna `x_n` por iteración retenida. |
| D. Ventana de período 3 | Localización del tramo con período 3 detectado, contraste con un caso caótico cercano y controles con distintos transitorios en los bordes. |
| E. Condiciones iniciales | Cuatro figuras por región. Los CSV reúnen trayectorias iniciales y tardías, diagnósticos por semilla y comparaciones por pares con alineación de ciclos. |

