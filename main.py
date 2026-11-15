import csv
import sys
import traceback
from decimal import Decimal, localcontext
from itertools import combinations

import dependencias

# Permite ejecutar con doble clic o desde un IDE: si falta alguna biblioteca, se instala antes de importarla.
try:
    dependencias.asegurar()
except Exception as error:
    dependencias.avisar("Error al preparar el entorno", str(error), error=True)
    sys.exit(1)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import configuracion as cfg  # noqa: E402
import graficos  # noqa: E402


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


def experimento_f():
    base = Decimal(cfg.SEMILLA_SENSIBILIDAD)
    n = np.arange(cfg.ITERACIONES_SENSIBILIDAD + 1)
    x = trayectoria(4, float(base), n[-1])
    casos, filas, distancias = [], [], []
    for k in cfg.CIFRAS_PERTURBADAS:
        # La cifra decimal k-ésima aumenta en una unidad; el resto de la semilla no cambia.
        semilla = base + Decimal(10) ** -k
        y = trayectoria(4, float(semilla), n[-1])
        distancia = np.abs(x - y)
        superan = np.flatnonzero(distancia > cfg.UMBRAL_SEPARACION)
        separacion = int(superan[0]) if len(superan) else None
        delta = abs(float(semilla) - float(base))
        filas.append((k, str(semilla), delta, separacion,
                      np.log2(cfg.UMBRAL_SEPARACION / delta)))
        distancias.extend((k, i, a, b, d) for i, a, b, d in zip(n, x, y, distancia))
        casos.append((k, str(semilla), delta, y, distancia, separacion))
    guardar_csv("f_separacion.csv", ("cifra_perturbada", "semilla", "delta_inicial_float64",
                                     "primera_iteracion_umbral", "prediccion_lyapunov_log2"), filas)
    guardar_csv("f_distancias.csv", ("cifra_perturbada", "n", "x_n", "y_n", "distancia"), distancias)
    k = np.array([c[0] for c in casos], dtype=float)
    separacion = np.array([c[5] for c in casos], dtype=float)
    pendiente, ordenada = np.polyfit(k, separacion, 1)
    guardar_csv("f_ajuste.csv", ("pendiente_iteraciones_por_cifra", "ordenada",
                                 "pendiente_teorica_log2_10", "lyapunov_estimado"),
                [(pendiente, ordenada, np.log2(10), np.log(10) / pendiente)])
    graficos.sensibilidad_trayectorias(FIGURAS / "f_trayectorias.png", str(base), x, casos)
    graficos.sensibilidad_distancias(FIGURAS / "f_distancias.png", casos, (pendiente, ordenada))


def semillas_muestra():
    # Malla determinista de puntos medios en (0, 1/2). Evita 0, 1/4, 1/2 y 3/4, y no incluye
    # pares simétricos s y 1 - s, que tendrían la misma imagen x_1.
    return (np.arange(cfg.N_MUESTRA) + 0.5) / (2 * cfg.N_MUESTRA)


def iterar(x, iteraciones):
    x = np.array(x, dtype=float)
    for _ in range(iteraciones):
        x = paso(4, x)
    return x


def resumen(muestra, teoricas):
    filas = [("media", muestra.mean(), *(d.mean() for d in teoricas)),
             ("varianza", muestra.var(ddof=1), *(d.var() for d in teoricas))]
    filas += [(f"cuantil_{p:.2f}", np.quantile(muestra, p), *(d.ppf(p) for d in teoricas))
              for p in cfg.CUANTILES]
    return filas


def ljung_box(serie, rezagos):
    n = len(serie)
    centrada = serie - serie.mean()
    acf = np.array([centrada[:-h] @ centrada[h:] for h in range(1, rezagos + 1)]) / (centrada @ centrada)
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, rezagos + 1)))
    return acf, q, stats.chi2.sf(q, rezagos)


def rachas(serie):
    # Wald–Wolfowitz: rachas por encima y por debajo de la mediana (se descartan los empates).
    signo = serie[serie != np.median(serie)] > np.median(serie)
    n1, n2 = float(signo.sum()), float((~signo).sum())
    n = n1 + n2
    observadas = 1 + np.count_nonzero(signo[1:] != signo[:-1])
    esperadas = 2 * n1 * n2 / n + 1
    varianza = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (observadas - esperadas) / np.sqrt(varianza)
    return observadas, esperadas, z, 2 * stats.norm.sf(abs(z))


def chi2_pares(serie, clases):
    # Pares disjuntos (y_1, y_2), (y_3, y_4), ... clasificados en clases equiprobables.
    cortes = np.quantile(serie, np.linspace(0, 1, clases + 1)[1:-1])
    clase = np.searchsorted(cortes, serie)
    m = len(clase) // 2 * 2
    tabla = np.zeros((clases, clases), dtype=int)
    np.add.at(tabla, (clase[:m:2], clase[1:m:2]), 1)
    estadistico, p, gl, _ = stats.chi2_contingency(tabla, correction=False)
    return estadistico, gl, p


def pruebas_dependencia(nombre, serie):
    acf, q, p_q = ljung_box(serie, cfg.REZAGOS_LJUNG_BOX)
    observadas, esperadas, z, p_r = rachas(serie)
    chi2, gl, p_c = chi2_pares(serie, cfg.CLASES_CHI2)
    filas = [
        (nombre, "autocorrelacion_rezago_1", acf[0], None, 2 * stats.norm.sf(abs(acf[0]) * np.sqrt(len(serie)))),
        (nombre, "ljung_box", q, cfg.REZAGOS_LJUNG_BOX, p_q),
        (nombre, "rachas_mediana_z", z, None, p_r),
        (nombre, "chi2_independencia_pares", chi2, gl, p_c),
    ]
    return acf, filas, (observadas, esperadas)


def experimento_g():
    semillas = semillas_muestra()
    x = iterar(semillas, cfg.ITERACION_OBSERVADA)
    if np.any((x <= 0) | (x >= 1)) or len(np.unique(x)) < len(x):
        raise RuntimeError("La muestra contiene valores degenerados o repetidos.")
    u = 2 / np.pi * np.arcsin(np.sqrt(x))
    guardar_csv("g_muestra.csv", ("i", "semilla", f"x_{cfg.ITERACION_OBSERVADA}", "u"),
                zip(range(1, len(x) + 1), semillas, x, u))

    uniforme, arcoseno = stats.uniform(), stats.beta(0.5, 0.5)
    guardar_csv("g_resumen.csv", ("estadistico", "muestra", "uniforme_0_1", "beta_1_2_1_2"),
                resumen(x, (uniforme, arcoseno)))
    ks = [("x", "uniforme_0_1", *stats.kstest(x, uniforme.cdf)[:2]),
          ("x", "beta_1_2_1_2", *stats.kstest(x, arcoseno.cdf)[:2])]

    # Serie temporal de una sola semilla, para contrastar con la muestra entre semillas.
    serie = iterar_serie(float(Decimal(cfg.SEMILLA_SENSIBILIDAD)))
    acf_m, filas_m, _ = pruebas_dependencia("muestra_entre_semillas", x)
    acf_s, filas_s, _ = pruebas_dependencia("trayectoria_unica", serie)
    guardar_csv("g_dependencia.csv", ("serie", "prueba", "estadistico", "grados_libertad", "p_valor"),
                filas_m + filas_s)
    guardar_csv("g_autocorrelaciones.csv", ("rezago", "muestra_entre_semillas", "trayectoria_unica"),
                zip(range(1, cfg.REZAGOS_LJUNG_BOX + 1), acf_m, acf_s))

    graficos.histograma(FIGURAS / "g_histograma.png", x, (uniforme, arcoseno), ks)
    graficos.distribucion_empirica(FIGURAS / "g_fda.png", x, (uniforme, arcoseno), ks)
    graficos.dependencia(FIGURAS / "g_dependencia.png", (x, serie), (acf_m, acf_s))
    return u, serie, ks


def iterar_serie(x0):
    x = trayectoria(4, x0, cfg.TRANSITORIO_SERIE + cfg.N_MUESTRA)
    return x[cfg.TRANSITORIO_SERIE + 1:]


def experimento_h(u, serie, ks):
    uniforme = stats.uniform()
    guardar_csv("h_resumen.csv", ("estadistico", "muestra_u", "uniforme_0_1"), resumen(u, (uniforme,)))
    ks = ks + [("u", "uniforme_0_1", *stats.kstest(u, uniforme.cdf)[:2])]
    guardar_csv("gh_kolmogorov_smirnov.csv", ("variable", "distribucion", "D", "p_valor"), ks)
    serie_u = 2 / np.pi * np.arcsin(np.sqrt(serie))
    graficos.uniformizacion(FIGURAS / "h_uniformizacion.png", u, serie_u, ks[-1])


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
    experimento_f()
    experimento_h(*experimento_g())


if __name__ == "__main__":
    try:
        print("Ejecutando los experimentos A-H (puede tardar alrededor de un minuto)...")
        main()
    except Exception:
        dependencias.avisar("Error durante la ejecución", traceback.format_exc(), error=True)
        sys.exit(1)
    dependencias.avisar("Ejecución completada",
                        f"Se generaron las figuras y las tablas en:\n{cfg.SALIDAS}")
    dependencias.abrir_carpeta(cfg.SALIDAS)
