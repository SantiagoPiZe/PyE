# Test Kolmogórov–Smirnov (seccion 2.7) y Ljung–Box (seccion 2.8)
import numpy as np
from scipy import stats  # type: ignore[reportMissingImports]

import configuracion as cfg


UNIFORME = stats.uniform(0, 1)
BETA_MEDIO = stats.beta(0.5, 0.5)

def decision(p_valor):
    return "Se rechaza H0" if p_valor < cfg.NIVEL_SIGNIFICACION else "No se rechaza H0"


def kolmogorov_smirnov(x, distribucion):
    resultado = stats.kstest(x, distribucion.cdf)
    return resultado.statistic, resultado.pvalue


def autocorrelaciones(x, rezagos):
    centrado = x - x.mean()
    denominador = np.sum(centrado ** 2)
    return np.array([np.sum(centrado[:-k] * centrado[k:]) / denominador
                     for k in range(1, rezagos + 1)])


def ljung_box(x, rezagos):
    n = len(x)
    rho = autocorrelaciones(x, rezagos)
    q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, rezagos + 1)))
    return q, stats.chi2.sf(q, rezagos)


def momentos_y_cuantiles(x):
    return x.mean(), x.var(ddof=1), np.quantile(x, cfg.CUANTILES)