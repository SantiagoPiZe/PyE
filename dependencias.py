"""Comprueba las bibliotecas necesarias y, si falta alguna, la instala con pip."""

import importlib.util
import subprocess
import sys
from pathlib import Path


MODULOS = ("numpy", "matplotlib", "scipy")
REQUISITOS = Path(__file__).resolve().parent / "requirements.txt"
# Con --sin-ventana no se abren ventanas ni carpetas (útil al ejecutar repetidamente desde la consola).
SIN_VENTANA = "--sin-ventana" in sys.argv


def asegurar():
    faltantes = [m for m in MODULOS if importlib.util.find_spec(m) is None]
    if not faltantes:
        return
    print(f"Faltan bibliotecas: {', '.join(faltantes)}. Se instalarán con pip (requiere internet)...")
    comando = [sys.executable, "-m", "pip", "install", "-r", str(REQUISITOS)]
    if subprocess.run(comando).returncode != 0:
        raise RuntimeError(
            "No se pudieron instalar las dependencias automáticamente.\n"
            f"Ejecute en una terminal:\n  \"{sys.executable}\" -m pip install -r \"{REQUISITOS}\""
        )
    importlib.invalidate_caches()
    print("Dependencias instaladas.\n")


def avisar(titulo, texto, error=False):
    """Muestra el mensaje en consola y en una ventana; la ventana persiste aunque la consola se cierre."""
    print(texto, file=sys.stderr if error else sys.stdout)
    if SIN_VENTANA:
        return
    try:
        import tkinter
        from tkinter import messagebox

        raiz = tkinter.Tk()
        raiz.withdraw()
        raiz.attributes("-topmost", True)
        (messagebox.showerror if error else messagebox.showinfo)(titulo, texto, parent=raiz)
        raiz.destroy()
    except Exception:
        # Sin entorno gráfico (por ejemplo, en un servidor): basta con la consola.
        pass


def abrir_carpeta(ruta):
    if SIN_VENTANA:
        return
    try:
        if sys.platform.startswith("win"):
            import os
            os.startfile(ruta)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(ruta)])
    except Exception:
        pass
