import csv
from decimal import Decimal, localcontext
from itertools import combinations

import numpy as np

import configuracion as cfg
import estadistica
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


# Cambia una sola cifra decimal de x0 (si la cifra es 9 resta 1 para no arrastrar). La usa el experimento F.
def semilla_perturbada(x0, cifra):
    digitos = list(f"{x0:.9f}"[2:])
    valor = int(digitos[cifra - 1])
    digitos[cifra - 1] = str(valor + 1 if valor < 9 else valor - 1)
    return float("0." + "".join(digitos))


# Aplica la recurrencia con r = 4 desde todas las semillas a la vez y devuelve el valor final. La usa el experimento G.
def observaciones(semillas, iteraciones):
    x = np.array(semillas, dtype=float)
    for _ in range(iteraciones):
        x = paso(cfg.R_CAOS, x)
    return x


# Experimento F: sensibilidad a la semilla con r = 4 (letra, sección 2; informe 3.2 y 4.2).
# Compara la trayectoria original con 9 semillas que difieren en una cifra decimal y registra cuándo la distancia supera 0,1.
def experimento_f():
    n = np.arange(cfg.ITERACIONES_SENSIBILIDAD + 1)
    x0 = cfg.SEMILLA_SENSIBILIDAD
    base = trayectoria(cfg.R_CAOS, x0, n[-1])
    perturbadas, distancias, tiempos, filas = [], [], [], []
    for cifra in cfg.CIFRAS_PERTURBADAS:
        y0 = semilla_perturbada(x0, cifra)
        serie = trayectoria(cfg.R_CAOS, y0, n[-1])
        d = np.abs(base - serie)
        superado = np.flatnonzero(d > cfg.UMBRAL_SEPARACION)
        tiempo = int(superado[0]) if len(superado) else None
        perturbadas.append((cifra, y0, serie))
        distancias.append((cifra, d))
        tiempos.append(tiempo)
        filas.append((cifra, f"{y0:.9f}", abs(y0 - x0), tiempo))
    cifras = np.array(cfg.CIFRAS_PERTURBADAS)
    observados = np.array([np.nan if t is None else t for t in tiempos])
    # Como la distancia se duplica en cada iteración, el tiempo de separación crece log2(10) por cifra.
    pendiente, intercepto = np.polyfit(cifras, observados, 1)
    ajustados = intercepto + pendiente * cifras
    r2 = 1 - np.sum((observados - ajustados) ** 2) / np.sum((observados - observados.mean()) ** 2)

    guardar_csv("f_sensibilidad.csv", ("cifra_modificada", "semilla_perturbada", "distancia_inicial",
                                       "primera_iteracion_sobre_umbral"), filas)
    guardar_csv("f_ajuste_tiempo.csv", ("pendiente_observada", "pendiente_teorica_log2_10",
                                        "intercepto", "r2"),
                [(pendiente, np.log2(10), intercepto, r2)])
    graficos.sensibilidad_trayectorias(FIGURAS / "f_trayectorias.png", n, base, perturbadas)
    graficos.sensibilidad_distancias(FIGURAS / "f_distancias.png", n, distancias, tiempos)
    graficos.tiempo_separacion(FIGURAS / "f_tiempo_separacion.png", cifras, observados,
                               pendiente, intercepto)

    print(f"\n[F] Sensibilidad (r = 4, x0 = {x0}): primera iteración con |x_n - y_n| > {cfg.UMBRAL_SEPARACION:g}")
    for cifra, _, _, tiempo in filas:
        print(f"    cifra {cifra}: n = {tiempo}")
    print(f"    pendiente = {pendiente:.2f} iteraciones por cifra (teoría log2(10) = {np.log2(10):.2f}), R2 = {r2:.3f}")


# Experimento G: distribución de x_1000 con N = 10 000 semillas k/(N+1) (letra, secciones 3.1 y 3.2; informe 3.3 y 4.3).
# Calcula media, varianza, cuantiles y el test de Kolmogórov–Smirnov frente a la uniforme y a la Beta(1/2, 1/2).
def experimento_g():
    semillas = np.arange(1, cfg.N_MUESTRA + 1) / (cfg.N_MUESTRA + 1)
    x = observaciones(semillas, cfg.ITERACION_OBSERVADA)
    media, varianza, cuantiles = estadistica.momentos_y_cuantiles(x)
    uniforme, beta = estadistica.UNIFORME, estadistica.BETA_MEDIO
    filas = [("media", media, uniforme.mean(), beta.mean()),
             ("varianza", varianza, uniforme.var(), beta.var())]
    filas += [(f"cuantil_{p:.2f}", q, uniforme.ppf(p), beta.ppf(p))
              for p, q in zip(cfg.CUANTILES, cuantiles)]
    guardar_csv("g_estadisticos.csv", ("medida", "muestra", "uniforme_0_1", "beta_1_2_1_2"), filas)
    ks = []
    for nombre, distribucion in (("Uniforme(0, 1)", uniforme), ("Beta(1/2, 1/2)", beta)):
        estadistico, p_valor = estadistica.kolmogorov_smirnov(x, distribucion)
        ks.append((nombre, estadistico, p_valor, estadistica.decision(p_valor)))
    guardar_csv("g_ks.csv", ("distribucion_teorica", "estadistico_D", "p_valor", "decision"), ks)
    graficos.distribucion(FIGURAS / "g_histograma.png", x)
    graficos.ecdf(FIGURAS / "g_ecdf.png", x)

    print(f"\n[G] Distribución de x_{cfg.ITERACION_OBSERVADA} con N = {len(x)} semillas k/(N+1)")
    print(f"    media = {media:.4f}, varianza = {varianza:.4f}  (Uniforme: 0.5, 0.0833; Beta: 0.5, 0.125)")
    for nombre, estadistico, p_valor, decision in ks:
        print(f"    KS frente a {nombre}: D = {estadistico:.4f}, p = {p_valor:.3g} -> {decision}")
    return x


# Aplica el test de Ljung-Box a una serie y calcula sus autocorrelaciones. La usan los experimentos H e I.
def ljung_box_serie(nombre, x):
    q, p_valor = estadistica.ljung_box(x, cfg.REZAGOS_LJUNG_BOX)
    fila = (nombre, q, cfg.REZAGOS_LJUNG_BOX, p_valor, estadistica.decision(p_valor))
    return fila, estadistica.autocorrelaciones(x, cfg.REZAGOS_LJUNG_BOX)


# Experimento H: dependencia con el test de Ljung-Box (letra, sección 3.3; informe 3.4 y 4.4).
# Se aplica a la muestra de x_1000 y a una sola trayectoria, y se grafica x_t contra x_t+1.
def experimento_h(muestra):
    trayectoria_unica = trayectoria(cfg.R_CAOS, cfg.SEMILLA, cfg.N_MUESTRA)[1:]
    fila_a, rho_a = ljung_box_serie("Muestra x_1000 ordenada por semilla", muestra)
    fila_b, rho_b = ljung_box_serie("Una sola trayectoria x_n (n = 1..N)", trayectoria_unica)
    guardar_csv("h_ljung_box.csv", ("serie", "estadistico_Q", "grados_libertad", "p_valor", "decision"),
                (fila_a, fila_b))
    guardar_csv("h_autocorrelaciones.csv", ("rezago", "muestra_por_semilla", "trayectoria_unica"),
                zip(range(1, cfg.REZAGOS_LJUNG_BOX + 1), rho_a, rho_b))
    graficos.dependencia(FIGURAS / "h_dependencia.png", [
        ("Muestra $x_{1000}$ ordenada por semilla", muestra, rho_a),
        (f"Una sola trayectoria ($x_0={cfg.SEMILLA:g}$)", trayectoria_unica, rho_b),
    ])

    print(f"\n[H] Ljung-Box (h = {cfg.REZAGOS_LJUNG_BOX}, alfa = {cfg.NIVEL_SIGNIFICACION})")
    for fila in (fila_a, fila_b):
        print(f"    {fila[0]}: Q = {fila[1]:.2f}, p = {fila[3]:.3g} -> {fila[4]}")
    return trayectoria_unica


# Experimento I: uniformización u = (2/pi) arcsin(sqrt(x)) (letra, sección 4; informe 3.5 y 4.5).
# Repite histograma, media, varianza, cuantiles, KS y Ljung-Box con los datos transformados.
def experimento_i(muestra, trayectoria_unica):
    u = 2 / np.pi * np.arcsin(np.sqrt(muestra))
    u_trayectoria = 2 / np.pi * np.arcsin(np.sqrt(trayectoria_unica))
    media, varianza, cuantiles = estadistica.momentos_y_cuantiles(u)
    uniforme = estadistica.UNIFORME
    filas = [("media", media, uniforme.mean()), ("varianza", varianza, uniforme.var())]
    filas += [(f"cuantil_{p:.2f}", q, uniforme.ppf(p)) for p, q in zip(cfg.CUANTILES, cuantiles)]
    guardar_csv("i_estadisticos.csv", ("medida", "muestra_transformada", "uniforme_0_1"), filas)
    estadistico, p_valor = estadistica.kolmogorov_smirnov(u, uniforme)
    guardar_csv("i_ks.csv", ("distribucion_teorica", "estadistico_D", "p_valor", "decision"),
                [("Uniforme(0, 1)", estadistico, p_valor, estadistica.decision(p_valor))])
    fila_a, _ = ljung_box_serie("Muestra u ordenada por semilla", u)
    fila_b, _ = ljung_box_serie("Una sola trayectoria u_n (n = 1..N)", u_trayectoria)
    guardar_csv("i_ljung_box.csv", ("serie", "estadistico_Q", "grados_libertad", "p_valor", "decision"),
                (fila_a, fila_b))
    graficos.uniformizacion(FIGURAS / "i_uniformizacion.png", u, u_trayectoria)

    print(f"\n[I] Datos transformados: media = {media:.4f}, varianza = {varianza:.4f} (Uniforme: 0.5, 0.0833)")
    print(f"    KS frente a Uniforme(0, 1): D = {estadistico:.4f}, p = {p_valor:.3g} -> {estadistica.decision(p_valor)}")
    for fila in (fila_a, fila_b):
        print(f"    Ljung-Box, {fila[0]}: Q = {fila[1]:.2f}, p = {fila[3]:.3g} -> {fila[4]}")


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
    muestra = experimento_g()
    trayectoria_unica = experimento_h(muestra)
    experimento_i(muestra, trayectoria_unica)
    print(f"Figuras y tablas guardadas en {cfg.SALIDAS}")


if __name__ == "__main__":
    main()