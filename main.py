import csv
from decimal import Decimal, localcontext
from itertools import combinations

import numpy as np

import configuracion as cfg
import graficos


FIGURAS = cfg.SALIDAS / "figuras"
TABLAS = cfg.SALIDAS / "tablas"
INICIO_TARDIO = cfg.TRANSITORIO + 1
TRAMO_TARDIO = slice(INICIO_TARDIO, INICIO_TARDIO + cfg.MUESTRA_TARDIA)
FIN_INICIAL = cfg.ITERACIONES_INICIALES + 1


def paso(r, x):
    return r * x * (1 - x)


def trayectoria(r, x0, iteraciones):
    valores = np.empty(iteraciones + 1)
    valores[0] = x0
    for n in range(iteraciones):
        valores[n + 1] = paso(r, valores[n])
    return valores


def bloque_tardio(parametros, transitorio, retenidos):
    parametros = np.asarray(parametros)
    x = np.full(len(parametros), cfg.SEMILLA)
    for _ in range(transitorio):
        x = paso(parametros, x)
    bloque = np.empty((retenidos, len(parametros)))
    for n in range(retenidos):
        x = paso(parametros, x)
        bloque[n] = x
    return bloque.T


def diagnostico(r, x):
    periodo, residual = None, None
    for p in range(1, min(cfg.PERIODO_MAXIMO, len(x) // 4) + 1):
        error = np.max(np.abs(x[p:] - x[:-p]))
        if error <= cfg.TOL_PERIODO:
            periodo, residual = p, error
            break
    ciclo = x[:periodo] if periodo else x[:0]
    with np.errstate(divide="ignore"):
        lyapunov = np.log(np.abs(r * (1 - 2 * x))).mean()
    return periodo, residual, ciclo, lyapunov


def etiqueta(r, d):
    periodo, _, ciclo, lyapunov = d
    if periodo == 1:
        return "Extinción" if ciclo[0] <= cfg.TOL_PERIODO else "Punto fijo"
    if periodo:
        nombre = "Ventana de período" if r > cfg.R_ACUMULACION else "Período"
        return f"{nombre} {periodo}"
    return "Caos (evidencia numérica)" if lyapunov > 0 else "Sin período detectado"


def guardar_csv(nombre, columnas, filas):
    with (TABLAS / nombre).open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo, lineterminator="\n")
        escritor.writerow(columnas)
        escritor.writerows(filas)


COLUMNAS = ("periodo_detectado", "residual_periodo", "ciclo_temporal",
            "ciclo_ordenado", "lyapunov_finito", "minimo", "maximo", "media")


def fila_diagnostico(d, x):
    periodo, residual, ciclo, lyapunov = d
    return (periodo, residual, " ".join(map(str, ciclo)), " ".join(map(str, sorted(ciclo))),
            lyapunov, x.min(), x.max(), x.mean())


def experimento_a(orbitas, diagnosticos):
    guardar_csv("a_trayectorias.csv", ("r", "x0", "n", "x_n"),
                ((r, cfg.SEMILLA, n, x) for r in orbitas
                 for n, x in enumerate(orbitas[r][cfg.SEMILLA])))
    filas, paneles = [], []
    for r, semillas in orbitas.items():
        x = semillas[cfg.SEMILLA]
        d = diagnosticos[r, cfg.SEMILLA]
        nombre = etiqueta(r, d)
        filas.append((r, cfg.SEMILLA, nombre, x[-1], *fila_diagnostico(d, x[TRAMO_TARDIO])))
        paneles.append((r, x[:FIN_INICIAL], nombre))
    guardar_csv("a_regimenes.csv", ("r", "x0", "regimen", "x_n_final", *COLUMNAS), filas)
    graficos.trayectorias(FIGURAS / "a_trayectorias.png", paneles)


def experimento_b():
    n = np.arange(cfg.ITERACIONES_EXPLICITA + 1)
    iterativa = trayectoria(4, cfg.SEMILLA, n[-1])
    angulo = np.arcsin(np.sqrt(cfg.SEMILLA))
    explicita = np.sin(np.ldexp(angulo, n)) ** 2
    explicita[0] = cfg.SEMILLA
    if cfg.SEMILLA in (0, 1):
        explicita[1:] = 0
    with localcontext() as contexto:
        contexto.prec = cfg.PRECISION_REFERENCIA
        referencia = [Decimal.from_float(cfg.SEMILLA)]
        for _ in n[1:]:
            referencia.append(paso(4, referencia[-1]))
        errores = [np.abs(iterativa - explicita)]
        for serie in (iterativa, explicita):
            errores.append(np.array([float(abs(Decimal.from_float(float(x)) - ref))
                                     for x, ref in zip(serie, referencia)]))
    guardar_csv("b_comparacion_r4.csv",
                ("n", "iterativa_float64", "explicita_float64", "referencia_decimal",
                 "error_entre_metodos", "error_iterativa_vs_ref", "error_explicita_vs_ref"),
                zip(n, iterativa, explicita, referencia, *errores))
    graficos.comparacion_r4(FIGURAS / "b_comparacion_r4.png", iterativa, explicita, errores)


def guardar_bifurcacion(nombre, parametros, bloque, transitorio, intervalo=None, *, zoom=False):
    indices = range(transitorio + 1, transitorio + 1 + bloque.shape[1])
    guardar_csv(nombre + ".csv", ("r", *(f"x_{n}" for n in indices)),
                ((r, *x) for r, x in zip(parametros, bloque)))
    graficos.bifurcacion(FIGURAS / (nombre + ".png"), parametros, bloque, transitorio, intervalo, zoom=zoom)


def experimento_c():
    parametros = np.linspace(*cfg.R_BIFURCACION, cfg.MALLA_BIFURCACION)
    bloque = bloque_tardio(parametros, cfg.TRANSITORIO, cfg.RETENIDOS_BIFURCACION)
    guardar_bifurcacion("c_bifurcacion", parametros, bloque, cfg.TRANSITORIO)


def experimento_d():
    parametros = np.linspace(*cfg.R_ZOOM, cfg.MALLA_ZOOM)
    bloque = bloque_tardio(parametros, cfg.TRANSITORIO_ZOOM, cfg.RETENIDOS_ZOOM)
    resultados = [diagnostico(r, x) for r, x in zip(parametros, bloque)]
    periodo_3 = [p == 3 and lyapunov < 0 for p, _, _, lyapunov in resultados]
    centro = int(np.argmin(np.abs(parametros - cfg.R_PERIODICO)))
    intervalo, bordes = None, []
    if periodo_3[centro]:
        izquierda = derecha = centro
        while izquierda > 0 and periodo_3[izquierda - 1]:
            izquierda -= 1
        while derecha + 1 < len(parametros) and periodo_3[derecha + 1]:
            derecha += 1
        intervalo = (parametros[izquierda], parametros[derecha])
        bordes = parametros[sorted({max(0, izquierda - 1), izquierda, derecha,
                                    min(len(parametros) - 1, derecha + 1)})]

    guardar_bifurcacion("c_bifurcacion_zoom", parametros, bloque, cfg.TRANSITORIO_ZOOM, intervalo, zoom=True)
    guardar_csv("d_diagnosticos_malla.csv", ("r", *COLUMNAS),
                ((r, *fila_diagnostico(d, x)) for r, x, d in zip(parametros, bloque, resultados)))
    controles = []
    for transitorio in cfg.TRANSITORIOS_CONTROL_BORDES:
        for r, x in zip(bordes, bloque_tardio(bordes, transitorio, cfg.RETENIDOS_ZOOM)):
            periodo = diagnostico(r, x)[0]
            controles.append((r, transitorio, periodo, np.max(np.abs(x[3:] - x[:-3]))))
    guardar_csv("d_control_bordes.csv", ("r", "transitorio", "periodo_detectado", "residual_p3"), controles)

    casos = (cfg.R_PERIODICO, cfg.R_CAOTICO_CERCANO)
    bloques = bloque_tardio(casos, cfg.TRANSITORIO_ZOOM, cfg.RETENIDOS_ZOOM)
    resultados = [diagnostico(r, x) for r, x in zip(casos, bloques)]
    guardar_csv("d_casos.csv", ("r", "x0", *COLUMNAS),
                ((r, cfg.SEMILLA, *fila_diagnostico(d, x)) for r, x, d in zip(casos, bloques, resultados)))
    guardar_csv("d_trayectorias.csv", ("r", "x0", "n", "x_n"),
                ((r, cfg.SEMILLA, n, valor) for r, x in zip(casos, bloques)
                 for n, valor in enumerate(x, cfg.TRANSITORIO_ZOOM + 1)))
    graficos.ventana(FIGURAS / "d_ventana_vs_caos.png", [
        (r, x, etiqueta(r, d)) for r, x, d in zip(casos, bloques, resultados)
    ])


def experimento_e(orbitas, diagnosticos):
    filas, pares, estados = [], [], []
    for r, semillas in orbitas.items():
        for x0, x in semillas.items():
            filas.append((r, x0, *fila_diagnostico(diagnosticos[r, x0], x[TRAMO_TARDIO])))
            estados.extend((r, x0, "inicial", n, valor) for n, valor in enumerate(x[:FIN_INICIAL]))
            estados.extend((r, x0, "tardio", n, valor)
                           for n, valor in enumerate(x[TRAMO_TARDIO], INICIO_TARDIO))
        for a, b in combinations(semillas, 2):
            diferencia = np.abs(semillas[a] - semillas[b])
            tarde = diferencia[TRAMO_TARDIO]
            ciclo_a = diagnosticos[r, a][2]
            ciclo_b = diagnosticos[r, b][2]
            fase, error = None, None
            if len(ciclo_a) and len(ciclo_a) == len(ciclo_b):
                errores = [np.max(np.abs(ciclo_a - np.roll(ciclo_b, -k))) for k in range(len(ciclo_a))]
                fase = int(np.argmin(errores))
                error = errores[fase]
            pares.append((r, a, b, diferencia[:FIN_INICIAL].mean(), tarde.mean(),
                          (tarde > cfg.TOL_PERIODO).mean(), fase, error))

    guardar_csv("e_diagnosticos_semillas.csv", ("r", "x0", *COLUMNAS), filas)
    guardar_csv("e_trayectorias.csv", ("r", "x0", "tramo", "n", "x_n"), estados)
    guardar_csv("e_comparaciones_pares.csv",
                ("r", "x0_a", "x0_b", "mae_inicial", "mae_tardio", "fraccion_tardia_diferente",
                 "desfase_b_adelantado", "error_maximo_alineado"), pares)
    for nombre, titulo, parametros in cfg.GRUPOS_SEMILLAS:
        paneles = [(r, orbitas[r]) for r in parametros if r in orbitas]
        if paneles:
            graficos.condiciones(FIGURAS / nombre, titulo, paneles)


def main():
    FIGURAS.mkdir(parents=True, exist_ok=True)
    TABLAS.mkdir(parents=True, exist_ok=True)
    total = max(cfg.ITERACIONES_INICIALES, cfg.TRANSITORIO + cfg.MUESTRA_TARDIA)
    orbitas = {r: {x0: trayectoria(r, x0, total) for x0 in cfg.SEMILLAS} for r in cfg.PARAMETROS}
    diagnosticos = {(r, x0): diagnostico(r, x[TRAMO_TARDIO])
                    for r, semillas in orbitas.items() for x0, x in semillas.items()}
    experimento_a(orbitas, diagnosticos)
    experimento_b()
    experimento_c()
    experimento_d()
    experimento_e(orbitas, diagnosticos)
    print(f"Figuras y tablas guardadas en {cfg.SALIDAS}")


if __name__ == "__main__":
    main()
