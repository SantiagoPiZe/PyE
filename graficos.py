
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
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#DDE5EB",
    "grid.linewidth": 0.65,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
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


GRIS = "#7A8790"
TEORICA = {"uniforme_0_1": ("Uniforme(0, 1)", CAOTICO, "--"),
           "beta_1_2_1_2": ("Beta(1/2, 1/2)", PERIODICO, "-")}


def _secuencial(n):
    return plt.get_cmap("Blues")(np.linspace(0.45, 1.0, n))


def sensibilidad_trayectorias(ruta, semilla, x, casos):
    filas = (len(casos) + 1) // 2
    figura, ejes = plt.subplots(filas, 2, figsize=(11.2, 1.55 * filas + 1.2), sharex=True,
                               sharey=True, squeeze=False, layout="constrained")
    figura.suptitle(f"Sensibilidad a la semilla en $r=4$  ·  $x_0={semilla}$ (gris) frente a $y_0$ (azul)")
    n = np.arange(len(x))
    for eje, (k, otra, _, y, _, separacion) in zip(ejes.flat, casos):
        eje.plot(n, x, color=GRIS, lw=0.9)
        eje.plot(n, y, color=COLOR, lw=0.9, ls="--")
        if separacion is not None:
            eje.axvline(separacion, color=CAOTICO, lw=1.0, ls=":")
        eje.set_title(f"$y_0={otra}$  ·  cifra {k}  ·  separación en $n={separacion}$",
                      fontsize=9, fontweight="normal", loc="left")
        eje.set(ylim=(-0.05, 1.05), yticks=(0, 1), xlim=(0, n[-1]))
    for eje in list(ejes.flat)[len(casos):]:
        eje.set_visible(False)
    figura.supxlabel("Iteración $n$")
    figura.supylabel("Estado")
    _guardar(figura, ruta)


def sensibilidad_distancias(ruta, casos, ajuste):
    figura, (izq, der) = plt.subplots(1, 2, figsize=(11.6, 5.0), width_ratios=(1.6, 1),
                                      layout="constrained")
    figura.suptitle("Separación de trayectorias cercanas en $r=4$")
    for (k, _, _, _, distancia, _), color in zip(casos, _secuencial(len(casos))):
        n = np.arange(len(distancia))
        izq.semilogy(n, np.ma.masked_less_equal(distancia, 0), color=color, lw=1.1,
                     label=f"$10^{{-{k}}}$")
    izq.axhline(cfg.UMBRAL_SEPARACION, color=CAOTICO, lw=1.0, ls="--",
                label=f"Umbral {cfg.UMBRAL_SEPARACION:g}")
    izq.set(title="Distancia $|x_n-y_n|$  ·  escala logarítmica", xlabel="Iteración $n$",
            ylabel="Distancia absoluta", xlim=(0, len(casos[0][4]) - 1), ylim=(1e-17, 2))
    izq.legend(title="Perturbación", fontsize=7.5, title_fontsize=8, ncols=2, frameon=False,
               loc="lower right")
    k = np.array([c[0] for c in casos])
    separacion = np.array([c[5] for c in casos])
    delta = np.array([c[2] for c in casos])
    pendiente, ordenada = ajuste
    der.plot(k, separacion, "o", color=COLOR, ms=7, label="Primera iteración con $|x_n-y_n|>0.1$")
    der.plot(k, pendiente * k + ordenada, color=COLOR, lw=1.2,
             label=f"Ajuste lineal: {pendiente:.2f} iteraciones por cifra")
    der.plot(k, np.log2(cfg.UMBRAL_SEPARACION / delta), color=CAOTICO, lw=1.2, ls="--",
             label=r"Predicción $\log_2(0.1/\delta_0)$, $\lambda=\ln 2$")
    der.set(title="Tiempo de separación", xlabel=r"Cifra perturbada $k$  ($\delta_0\approx10^{-k}$)",
            ylabel="Iteración", xticks=k[::2])
    der.legend(frameon=False, fontsize=8.5, loc="upper left")
    _guardar(figura, ruta)


def _ks_texto(ks):
    return "\n".join(f"KS vs {TEORICA[d][0]}: D={D:.4f}, p={p:.3g}" for _, d, D, p in ks)


def histograma(ruta, x, distribuciones, ks):
    figura, eje = plt.subplots(figsize=(9.6, 5.6), layout="constrained")
    figura.suptitle(f"Distribución de $x_{{{cfg.ITERACION_OBSERVADA}}}$ en $r=4$  ·  "
                    f"{len(x):,} semillas deterministas")
    eje.hist(x, bins=cfg.INTERVALOS_HISTOGRAMA, range=(0, 1), density=True, color="#B9CCD8",
             edgecolor="white", linewidth=1.0, label=f"Histograma ({cfg.INTERVALOS_HISTOGRAMA} intervalos)")
    malla = np.linspace(0.002, 0.998, 800)
    for d, clave in zip(distribuciones, TEORICA):
        nombre, color, estilo = TEORICA[clave]
        eje.plot(malla, d.pdf(malla), color=color, lw=2, ls=estilo, label=f"Densidad {nombre}")
    eje.text(0.5, 0.97, _ks_texto(ks), transform=eje.transAxes, ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="#DDE5EB"))
    eje.set(xlabel=f"$x_{{{cfg.ITERACION_OBSERVADA}}}$", ylabel="Densidad", xlim=(0, 1), ylim=(0, 4.2))
    eje.legend(frameon=False, loc="center", bbox_to_anchor=(0.5, 0.55))
    _guardar(figura, ruta)


def distribucion_empirica(ruta, x, distribuciones, ks):
    figura, eje = plt.subplots(figsize=(8.4, 6.0), layout="constrained")
    figura.suptitle("Función de distribución empírica y teóricas")
    ordenados = np.sort(x)
    fn = np.arange(1, len(x) + 1) / len(x)
    eje.step(ordenados, fn, where="post", color=COLOR, lw=1.6, label="Empírica $F_n$")
    malla = np.linspace(0, 1, 800)
    for d, clave in zip(distribuciones, TEORICA):
        nombre, color, estilo = TEORICA[clave]
        eje.plot(malla, d.cdf(malla), color=color, lw=1.4, ls=estilo, label=nombre)
    # Marcar la distancia máxima frente a la uniforme.
    i = int(np.argmax(np.maximum(fn - ordenados, ordenados - (fn - 1 / len(x)))))
    eje.annotate("", xy=(ordenados[i], ordenados[i]), xytext=(ordenados[i], fn[i]),
                 arrowprops=dict(arrowstyle="<->", color="#293844", lw=1.1))
    eje.text(ordenados[i] - 0.02, (ordenados[i] + fn[i]) / 2, f"$D$ vs uniforme = {ks[0][2]:.3f}",
             fontsize=9, va="center", ha="right")
    eje.set(xlabel="$x$", ylabel="$F(x)$", xlim=(0, 1), ylim=(0, 1))
    eje.legend(frameon=False, loc="upper left")
    _guardar(figura, ruta)


def _acf(eje, acf, n, color, titulo):
    rezagos = np.arange(1, len(acf) + 1)
    banda = 1.96 / np.sqrt(n)
    eje.axhspan(-banda, banda, color=GRIS, alpha=0.18, label=r"$\pm1.96/\sqrt{N}$")
    eje.vlines(rezagos, 0, acf, color=color, lw=2)
    eje.plot(rezagos, acf, "o", color=color, ms=4)
    eje.axhline(0, color="#293844", lw=0.8)
    eje.set(title=titulo, xlabel="Rezago $h$", ylabel=r"$\hat\rho(h)$", ylim=(-0.1, 0.1),
            xticks=rezagos[::2])
    eje.legend(frameon=False, fontsize=8, loc="upper right")


def dependencia(ruta, series, acfs):
    figura, ejes = plt.subplots(2, 2, figsize=(11.0, 9.0), width_ratios=(1, 1.25), layout="constrained")
    figura.suptitle("Dependencia entre observaciones consecutivas")
    titulos = ("Muestra entre semillas", "Trayectoria única")
    etiquetas = (("$y_i$", "$y_{i+1}$"), ("$x_n$", "$x_{n+1}$"))
    for fila, serie, acf, titulo, (ex, ey), color in zip(ejes, series, acfs, titulos, etiquetas,
                                                          (COLOR, CAOTICO)):
        fila[0].plot(serie[:-1], serie[1:], ".", color=color, ms=1.6, alpha=0.5)
        fila[0].set(title=titulo, xlabel=ex, ylabel=ey, xlim=(0, 1), ylim=(0, 1), aspect="equal")
        _acf(fila[1], acf, len(serie), color, "Autocorrelación muestral")
    _guardar(figura, ruta)


def uniformizacion(ruta, u, serie_u, ks):
    figura, (hist, fda, tienda) = plt.subplots(1, 3, figsize=(14.0, 4.8), width_ratios=(1.25, 1, 1),
                                               layout="constrained")
    figura.suptitle(r"Uniformización $u=\frac{2}{\pi}\arcsin\sqrt{x}$")
    hist.hist(u, bins=cfg.INTERVALOS_HISTOGRAMA, range=(0, 1), density=True, color="#B9CCD8",
              edgecolor="white", linewidth=1.0, label="Histograma de $u$")
    hist.plot((0, 1), (1, 1), color=CAOTICO, lw=2, ls="--", label="Densidad Uniforme(0, 1)")
    hist.set(title=f"KS vs uniforme: D={ks[2]:.4f}, p={ks[3]:.3g}", xlabel="$u$", ylabel="Densidad",
             xlim=(0, 1), ylim=(0, 1.5))
    hist.legend(frameon=False, loc="upper center", fontsize=9)
    ordenados = np.sort(u)
    fda.step(ordenados, np.arange(1, len(u) + 1) / len(u), where="post", color=COLOR, lw=1.6,
             label="Empírica $F_n$")
    fda.plot((0, 1), (0, 1), color=CAOTICO, lw=1.3, ls="--", label="Uniforme(0, 1)")
    fda.set(title="Función de distribución", xlabel="$u$", ylabel="$F(u)$", xlim=(0, 1), ylim=(0, 1))
    fda.legend(frameon=False, loc="upper left", fontsize=9)
    tienda.plot(serie_u[:-1], serie_u[1:], ".", color=CAOTICO, ms=1.6, alpha=0.5)
    tienda.set(title="Trayectoria única transformada", xlabel="$u_n$", ylabel="$u_{n+1}$",
               xlim=(0, 1), ylim=(0, 1), aspect="equal")
    _guardar(figura, ruta)
