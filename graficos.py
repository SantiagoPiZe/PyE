
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import configuracion as cfg


COLOR = "#1E5674"
PERIODICO = "#107264"
CAOTICO = "#A13E30"
SEMILLA_COLORES = ("#1D5E88", "#A23C2A", "#187A59", "#6C4391")
SEMILLA_MARCAS = ("o", "s", "^", "D")
SEMILLA_LINEAS = ("-", "--", "-.", ":")

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDE5EB",
    "grid.linewidth": 0.65,
    "figure.titlesize": 16,
    "figure.titleweight": "semibold",
    "savefig.facecolor": "white",
})


def _guardar(figura, ruta):
    # Escribir el PNG completo antes de sustituir el archivo anterior.
    temporal = None
    try:
        with NamedTemporaryFile(dir=ruta.parent, suffix=".png", delete=False) as archivo:
            temporal = Path(archivo.name)
            figura.savefig(archivo, format="png", dpi=cfg.DPI)
        temporal.replace(ruta)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)
        plt.close(figura)


def trayectorias(ruta, muestras):
    filas = (len(muestras) + 1) // 2
    figura, ejes = plt.subplots(filas, 2, figsize=(10.6, 2.25 * filas + 0.65),
                               sharex=True, sharey=True, squeeze=False, layout="constrained")
    figura.suptitle(f"Trayectorias del mapa logístico  ·  $x_0={cfg.SEMILLA:g}$")
    for eje, (r, orbita, etiqueta) in zip(ejes.flat, muestras):
        eje.plot(np.arange(len(orbita)), orbita, color=COLOR, lw=0.95)
        eje.scatter(0, orbita[0], color=COLOR, s=16, zorder=3)
        eje.set(title=f"$r={r:g}$  ·  {etiqueta}", ylim=(-0.035, 1.035),
                xlim=(-1, cfg.ITERACIONES_INICIALES), yticks=(0, 0.5, 1))
    for eje in list(ejes.flat)[len(muestras):]:
        eje.set_visible(False)
    figura.supxlabel("Iteración $n$")
    figura.supylabel("Estado $x_n$")
    _guardar(figura, ruta)


def comparacion_r4(ruta, iterativa, explicita, errores):
    n = np.arange(len(iterativa))
    fin = cfg.N_VALIDACION_EXPLICITA + 1
    figura, (arriba, abajo) = plt.subplots(2, 1, figsize=(9.6, 7.4), layout="constrained")
    figura.suptitle(f"Recurrencia y solución explícita en $r=4$  ·  $x_0={cfg.SEMILLA:g}$")
    arriba.plot(n[:fin], iterativa[:fin], "o-", color=COLOR, label="Recurrencia float64")
    arriba.plot(n[:fin], explicita[:fin], "s", mfc="none", color=CAOTICO,
                markersize=9, label="Fórmula explícita float64")
    arriba.set(title="Coincidencia en las primeras iteraciones", ylabel="Estado $x_n$",
               xlabel="Iteración $n$", xticks=n[:fin:2])
    arriba.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    arriba.set_ylim(-0.05, 1.22)
    for error, etiqueta, color, estilo in zip(
        errores, ("Entre métodos float64", "Recurrencia vs. referencia decimal",
                  "Fórmula vs. referencia decimal"),
        ("#293844", COLOR, CAOTICO), ("-", "--", ":"),
    ):
        abajo.semilogy(n, np.ma.masked_less_equal(error, 0), linestyle=estilo,
                       lw=1.35, color=color, label=etiqueta)
    abajo.axhline(cfg.UMBRAL_ERROR_EXPLICITA, color="#6E6E6E", lw=0.9, ls="-.",
                  label=f"Umbral descriptivo ({cfg.UMBRAL_ERROR_EXPLICITA:g})")
    abajo.axvspan(0, fin - 1, alpha=0.11, color=COLOR)
    abajo.set(title="Errores absolutos  ·  escala logarítmica", xlabel="Iteración $n$",
              ylabel="Error absoluto", xlim=(0, n[-1]))
    abajo.legend(frameon=False, loc="lower right", fontsize=9)
    _guardar(figura, ruta)


def bifurcacion(ruta, parametros, bloque, transitorio, intervalo=None, *, zoom=False):
    figura, eje = plt.subplots(figsize=(10.8, 6.1), layout="constrained")
    figura.suptitle("Ventana de período 3 en el mapa logístico" if zoom else
                    "Diagrama de bifurcación del mapa logístico")
    eje.set_title(f"$x_0={cfg.SEMILLA:g}$  ·  {len(parametros):,} valores de $r$  ·  "
                  f"{transitorio:,} iteraciones de transitorio y {bloque.shape[1]} puntos retenidos",
                  fontsize=10, fontweight="normal")
    if zoom:
        if intervalo is not None:
            eje.axvspan(*intervalo, color=PERIODICO, alpha=0.15, label="Tramo detectado de período 3")
        eje.axvline(cfg.R_PERIODICO, color=PERIODICO, lw=1.15, ls="--", zorder=3,
                    label=f"Caso periódico ($r={cfg.R_PERIODICO:g}$)")
        eje.axvline(cfg.R_CAOTICO_CERCANO, color=CAOTICO, lw=1.15, ls=":", zorder=3,
                    label=f"Caso caótico ($r={cfg.R_CAOTICO_CERCANO:g}$)")
        eje.legend(framealpha=0.95, loc="lower center", fontsize=9, ncols=3)
    else:
        eje.axvline(3, color=PERIODICO, lw=0.9, ls="--", label="Primera bifurcación ($r=3$)")
        eje.axvline(cfg.R_ACUMULACION, color=CAOTICO, lw=0.9, ls=":",
                    label=f"Acumulación ($r^*\u2248{cfg.R_ACUMULACION:g}$)")
        eje.legend(framealpha=0.95, loc="lower left", fontsize=9)
    eje.plot(np.repeat(parametros, bloque.shape[1]), bloque.ravel(), ".",
             color="#193449", ms=0.55 if zoom else 0.43, alpha=0.23, zorder=2)
    eje.set(xlabel="Parámetro $r$", ylabel="Estados retenidos $x_n$",
            xlim=(parametros[0], parametros[-1]), ylim=(-0.018, 1.025))
    _guardar(figura, ruta)


def ventana(ruta, muestras):
    figura, ejes = plt.subplots(2, 2, figsize=(11.2, 6.2), sharex=True,
                               width_ratios=(1.55, 1), layout="constrained")
    figura.suptitle(f"Órbitas tardías cerca de la ventana de período 3  ·  $x_0={cfg.SEMILLA:g}$")
    for (eje, residual), (r, bloque, etiqueta), color in zip(ejes, muestras, (PERIODICO, CAOTICO)):
        valores = bloque[:cfg.PUNTOS_TARDIOS_FIGURA]
        n = cfg.TRANSITORIO_ZOOM + 1 + np.arange(len(valores))
        eje.plot(n, valores, "o-", color=color, markersize=3.4, lw=1.1)
        eje.set(title=f"$r={r:g}$  ·  {etiqueta}", ylim=(0, 1), ylabel="Estado $x_n$")
        residual.plot(n[3:], np.abs(valores[3:] - valores[:-3]), "o-", color=color, markersize=3.4, lw=1.1)
        residual.set(title="Diferencia a tres iteraciones", ylim=(-0.035, 1),
                     ylabel="$|x_n-x_{n-3}|$")
    figura.supxlabel("Iteración $n$")
    _guardar(figura, ruta)


def condiciones(ruta, grupo, muestras):
    figura, ejes = plt.subplots(len(muestras), 2, figsize=(11.2, 2.25 * len(muestras) + 1.55),
                               sharex="col", sharey=True, squeeze=False, layout="constrained")
    figura.suptitle(f"Influencia de la condición inicial  ·  {grupo}")
    temprano = np.arange(cfg.PUNTOS_TARDIOS_FIGURA + 1)
    tardio = cfg.TRANSITORIO + 1 + np.arange(cfg.PUNTOS_TARDIOS_FIGURA)
    for fila, (r, trayectorias_semillas) in enumerate(muestras):
        for indice, (semilla, orbita) in enumerate(trayectorias_semillas.items()):
            for eje, n in zip(ejes[fila], (temprano, tardio)):
                eje.plot(n, orbita[n], color=SEMILLA_COLORES[indice % len(SEMILLA_COLORES)],
                         linestyle=SEMILLA_LINEAS[indice % len(SEMILLA_LINEAS)],
                         marker=SEMILLA_MARCAS[indice % len(SEMILLA_MARCAS)], lw=1.05,
                         markerfacecolor="white", markersize=4, markevery=(indice % 4, 4),
                         label=f"$x_0={semilla:g}$")
        for eje, etiqueta in zip(ejes[fila], ("inicio", "después del transitorio")):
            eje.set(title=f"$r={r:g}$  ·  {etiqueta}", ylim=(-0.045, 1.045))
    handles, etiquetas = ejes[0, 0].get_legend_handles_labels()
    figura.legend(handles, etiquetas, loc="outside lower center", ncols=len(etiquetas), frameon=False)
    for eje in ejes[-1]:
        eje.set_xlabel("Iteración $n$")
    figura.supylabel("Estado $x_n$")
    _guardar(figura, ruta)

#Experimenti de sensibilidad
def sensibilidad_trayectorias(ruta, n, base, perturbadas):
    filas = (len(perturbadas) + 2) // 3
    figura, ejes = plt.subplots(filas, 3, figsize=(11.2, 2.25 * filas + 1.2), sharex=True,
                               sharey=True, squeeze=False, layout="constrained")
    figura.suptitle(f"Trayectorias con semillas próximas  ·  $r={cfg.R_CAOS:g}$  ·  "
                    f"$x_0={cfg.SEMILLA_SENSIBILIDAD:.9f}$")
    for eje, (cifra, y0, serie) in zip(ejes.flat, perturbadas):
        eje.plot(n, base, color=COLOR, lw=1.0, label="$x_n$ (semilla original)")
        eje.plot(n, serie, color=CAOTICO, lw=1.0, ls="--", label="$y_n$ (semilla perturbada)")
        eje.set(title=f"$y_0={y0:.9f}$  ·  $\\Delta_0=10^{{-{cifra}}}$", ylim=(-0.03, 1.03),
                xlim=(0, n[-1]), yticks=(0, 0.5, 1))
    for eje in list(ejes.flat)[len(perturbadas):]:
        eje.set_visible(False)
    handles, etiquetas = ejes[0, 0].get_legend_handles_labels()
    figura.legend(handles, etiquetas, loc="outside lower center", ncols=2, frameon=False)
    for eje in ejes[-1]:
        eje.set_xlabel("Iteración $n$")
    figura.supylabel("Estado")
    _guardar(figura, ruta)


def sensibilidad_distancias(ruta, n, distancias, tiempos):
    figura, eje = plt.subplots(figsize=(10.2, 5.8), layout="constrained")
    figura.suptitle(f"Distancia entre trayectorias  ·  $r={cfg.R_CAOS:g}$")
    colores = plt.cm.viridis(np.linspace(0.05, 0.92, len(distancias)))
    for (cifra, d), tiempo, color in zip(distancias, tiempos, colores):
        eje.semilogy(n, np.ma.masked_less_equal(d, 0), color=color, lw=1.1,
                     label=f"$\\Delta_0=10^{{-{cifra}}}$")
        if tiempo is not None:
            eje.plot(tiempo, d[tiempo], "o", color=color, markersize=5, zorder=4)
    eje.axhline(cfg.UMBRAL_SEPARACION, color="#444444", lw=1.0, ls="-.",
                label=f"Umbral ({cfg.UMBRAL_SEPARACION:g})")
    eje.set(xlabel="Iteración $n$", ylabel="$|x_n-y_n|$", xlim=(0, n[-1]))
    eje.legend(frameon=False, ncols=2, fontsize=9, loc="lower right")
    _guardar(figura, ruta)


def tiempo_separacion(ruta, cifras, tiempos, pendiente, intercepto):
    figura, eje = plt.subplots(figsize=(7.6, 5.0), layout="constrained")
    figura.suptitle("Tiempo de separación según la proximidad inicial")
    cifras = np.asarray(cifras, dtype=float)
    malla = np.linspace(cifras.min(), cifras.max(), 50)
    eje.plot(malla, np.log2(10) * (malla - 1), color=PERIODICO, lw=1.3, ls="--",
             label="Teoría: $(k-1)\\,\\log_2 10$")
    eje.plot(malla, intercepto + pendiente * malla, color=CAOTICO, lw=1.3, ls=":",
             label=f"Ajuste lineal (pendiente {pendiente:.2f})")
    eje.plot(cifras, tiempos, "o", color=COLOR, markersize=6, label="Observado")
    eje.set(xlabel="Cifra decimal modificada $k$  ($\\Delta_0=10^{-k}$)",
            ylabel="Primera iteración con $|x_n-y_n|>0{,}1$")
    eje.legend(frameon=False, loc="upper left")
    _guardar(figura, ruta)


def distribucion(ruta, x):
    figura, ejes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True, layout="constrained")
    figura.suptitle(f"Distribución de $x_{{{cfg.ITERACION_OBSERVADA}}}$  ·  $N={len(x)}$  ·  "
                    f"{cfg.INTERVALOS_HISTOGRAMA} intervalos")
    t = np.linspace(0.0015, 0.9985, 1500)
    casos = (("Frente a la Uniforme(0, 1)", np.ones_like(t), PERIODICO, "Densidad uniforme"),
             ("Frente a la Beta(1/2, 1/2)", 1 / (np.pi * np.sqrt(t * (1 - t))), CAOTICO,
              "Densidad Beta(1/2, 1/2)"))
    for eje, (titulo, densidad, color, etiqueta) in zip(ejes, casos):
        eje.hist(x, bins=cfg.INTERVALOS_HISTOGRAMA, range=(0, 1), density=True,
                 color="#9DB8C9", edgecolor="white", label="Histograma de densidad")
        eje.plot(t, densidad, color=color, lw=2.0, label=etiqueta)
        eje.set(title=titulo, xlabel=f"$x_{{{cfg.ITERACION_OBSERVADA}}}$", xlim=(0, 1), ylim=(0, 4.6))
        eje.legend(frameon=False, loc="upper center", fontsize=9)
    ejes[0].set_ylabel("Densidad")
    _guardar(figura, ruta)


def ecdf(ruta, x):
    n = len(x)
    xs = np.sort(x)
    empirica = np.arange(1, n + 1) / n
    uniforme = xs
    beta = 2 / np.pi * np.arcsin(np.sqrt(xs))
    figura, (arriba, abajo) = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True,
                                           height_ratios=(2, 1), layout="constrained")
    figura.suptitle("Función de distribución empírica y teóricas")
    arriba.plot(xs, empirica, color=COLOR, lw=1.8, label="Empírica")
    arriba.plot(xs, uniforme, color=PERIODICO, lw=1.3, ls="--", label="Uniforme(0, 1)")
    arriba.plot(xs, beta, color=CAOTICO, lw=1.3, ls=":", label="Beta(1/2, 1/2)")
    arriba.set(ylabel="$F(x)$", ylim=(0, 1.02), xlim=(0, 1))
    arriba.legend(frameon=False, loc="lower right")
    abajo.plot(xs, empirica - uniforme, color=PERIODICO, lw=1.2, ls="--", label="Empírica $-$ Uniforme")
    abajo.plot(xs, empirica - beta, color=CAOTICO, lw=1.2, ls=":", label="Empírica $-$ Beta(1/2, 1/2)")
    abajo.axhline(0, color="#444444", lw=0.8)
    abajo.set(xlabel=f"$x_{{{cfg.ITERACION_OBSERVADA}}}$", ylabel="Diferencia")
    abajo.legend(frameon=False, loc="lower center", ncols=2, fontsize=9)
    _guardar(figura, ruta)


def dependencia(ruta, series):
    figura, ejes = plt.subplots(2, len(series), figsize=(5.7 * len(series), 7.6),
                               squeeze=False, layout="constrained")
    figura.suptitle("Dependencia entre observaciones consecutivas")
    for columna, (titulo, x, acf) in enumerate(series):
        arriba, abajo = ejes[0, columna], ejes[1, columna]
        arriba.plot(x[:-1], x[1:], ".", color=COLOR, markersize=1.6, alpha=0.35)
        arriba.set(title=titulo, xlabel="$x_t$", ylabel="$x_{t+1}$", xlim=(0, 1), ylim=(0, 1))
        rezagos = np.arange(1, len(acf) + 1)
        cota = 1.96 / np.sqrt(len(x))
        abajo.bar(rezagos, acf, color=COLOR, width=0.6)
        abajo.axhline(cota, color=CAOTICO, lw=1.0, ls="--", label="Cotas $\\pm 1{,}96/\\sqrt{N}$")
        abajo.axhline(-cota, color=CAOTICO, lw=1.0, ls="--")
        abajo.axhline(0, color="#444444", lw=0.8)
        abajo.set(xlabel="Rezago", ylabel="Autocorrelación", xticks=rezagos[::2],
                  ylim=(-0.12, 0.12))
        abajo.legend(frameon=False, loc="upper right", fontsize=9)
    _guardar(figura, ruta)

# Uniformizacion de los datos
def uniformizacion(ruta, u, par_trayectoria):
    n = len(u)
    us = np.sort(u)
    figura, ejes = plt.subplots(1, 3, figsize=(15.0, 4.6), layout="constrained")
    figura.suptitle(f"Datos transformados  $u=\\frac{{2}}{{\\pi}}\\arcsin\\sqrt{{x}}$  ·  $N={n}$")
    ejes[0].hist(u, bins=cfg.INTERVALOS_HISTOGRAMA, range=(0, 1), density=True,
                 color="#9DB8C9", edgecolor="white", label="Histograma de densidad")
    ejes[0].axhline(1, color=PERIODICO, lw=2.0, label="Densidad uniforme")
    ejes[0].set(title="Histograma", xlabel="$u$", ylabel="Densidad", xlim=(0, 1), ylim=(0, 1.6))
    ejes[0].legend(frameon=False, loc="upper center", fontsize=9)
    ejes[1].plot(us, np.arange(1, n + 1) / n - us, color=COLOR, lw=1.3)
    ejes[1].axhline(0, color="#444444", lw=0.8)
    ejes[1].set(title="Empírica $-$ Uniforme", xlabel="$u$", ylabel="$F_N(u)-u$", xlim=(0, 1))
    ejes[2].plot(par_trayectoria[:-1], par_trayectoria[1:], ".", color=CAOTICO, markersize=1.6, alpha=0.35)
    ejes[2].set(title="Trayectoria única transformada", xlabel="$u_t$", ylabel="$u_{t+1}$",
                xlim=(0, 1), ylim=(0, 1))
    _guardar(figura, ruta)
