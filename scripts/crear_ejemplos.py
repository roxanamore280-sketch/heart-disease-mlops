"""Genera los CSV de ejemplo para la inferencia batch y la API.

Uso, desde la raíz del proyecto:

    uv run python scripts/crear_ejemplos.py

- pacientes_validos.csv: 5 filas del conjunto de TEST (misma partición que
  train.py: 80/20, semilla 42), sin la columna `target`. Se toman de test para
  no "predecir" pacientes con los que el modelo ya se ha entrenado. Incluye una
  fila con `ca` vacío para mostrar que el pipeline imputa los ausentes.
- pacientes_invalidos.csv: copia de esas filas con errores introducidos a
  propósito (un valor por fila), para demostrar los mensajes de error.
- pacientes_sin_columna.csv: falta la columna `thal` (error de estructura).

No modifica el dataset original.
"""

import numpy as np

from mlops_course.config import load_config, resolve_path
from mlops_course.data import FEATURES, load_data, split_data

cfg = load_config()
df = load_data(resolve_path(cfg["datos"]["ruta"]))
_, _, test = split_data(
    df, cfg["datos"]["test_size"], cfg["datos"]["val_size"], cfg["semilla"]
)

validos = test[FEATURES].head(5).reset_index(drop=True)
validos.loc[4, "ca"] = np.nan  # ausente permitido: el pipeline lo imputa

invalidos = validos.copy().astype("object")
invalidos.loc[0, "thal"] = 5  # código no admitido (válidos: 3, 6, 7)
invalidos.loc[1, "age"] = -3  # fuera de rango
invalidos.loc[2, "chol"] = "alto"  # texto en una variable numérica
invalidos.loc[3, "sex"] = np.nan  # obligatorio vacío
# la fila 4 se deja correcta: el resto del lote se sigue prediciendo

carpeta = resolve_path("data/ejemplos")
carpeta.mkdir(parents=True, exist_ok=True)
validos.to_csv(carpeta / "pacientes_validos.csv", index=False)
invalidos.to_csv(carpeta / "pacientes_invalidos.csv", index=False)
validos.drop(columns="thal").to_csv(carpeta / "pacientes_sin_columna.csv", index=False)
print(f"Ejemplos creados en {carpeta}")
