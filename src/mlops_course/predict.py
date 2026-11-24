"""Carga del modelo y predicción (en memoria y batch sobre CSV).

Uso batch, desde la raíz del proyecto:

    uv run python -m mlops_course.predict
    uv run python -m mlops_course.predict --input data/ejemplos/pacientes_invalidos.csv

El batch reutiliza el pipeline guardado en `artifacts/model.joblib` (no reentrena).
La salida conserva las filas de entrada en el mismo orden y añade:
`probabilidad` (de la clase 1), `prediccion` (0 sano / 1 enfermo), `error`,
`umbral` y `model_version`.
"""

import argparse
import csv
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlops_course.config import load_config, resolve_path, tracking_uri
from mlops_course.data import CATEGORIAS, FEATURES, NUMERICAS, TARGET
from mlops_course.schemas import RANGOS_NUMERICOS  # mismos límites que la API

# Variables que pueden venir vacías: el pipeline las imputa (como en entrenamiento).
OPCIONALES = ["ca", "thal"]


# Registro de inferencias: SOLO metadatos de cada ejecución, nunca valores clínicos.
CAMPOS_REGISTRO = [
    "fecha",
    "origen",
    "model_version",
    "n_registros",
    "n_predichos",
    "n_errores",
    "tipo_error",
    "campos_con_error",
]


def registrar_inferencia(
    cfg: dict,
    origen: str,
    model_version: str | None,
    n_registros: int,
    n_errores: int,
    tipo_error: str = "",
    campos_con_error: list[str] | None = None,
) -> None:
    """Añade una fila a `logs/inferencias.csv` (ruta en cfg["api"]["inferencias"]).

    Guarda fecha, origen (batch/api), versión del modelo, nº de registros y de
    errores, y el TIPO de error y los NOMBRES de los campos afectados. No guarda
    ningún valor de entrada: la edad o el colesterol de un paciente no deben
    quedar en un log. Si el registro falla, avisa pero no interrumpe la predicción.
    """
    ruta = resolve_path(cfg["api"]["inferencias"])
    fila = {
        "fecha": datetime.now(UTC).isoformat(timespec="seconds"),
        "origen": origen,
        "model_version": model_version or "",
        "n_registros": n_registros,
        "n_predichos": max(n_registros - n_errores, 0),
        "n_errores": n_errores,
        "tipo_error": tipo_error,
        "campos_con_error": ";".join(sorted(set(campos_con_error or []))),
    }
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not ruta.exists()
        with ruta.open("a", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=CAMPOS_REGISTRO)
            if nuevo:
                escritor.writeheader()
            escritor.writerow(fila)
    except OSError as e:
        warnings.warn(
            f"No se pudo escribir el registro de inferencias: {e}", stacklevel=2
        )


def load_model(cfg: dict) -> tuple[Any, dict]:
    """Carga el pipeline local y sus metadatos desde las rutas de la configuración.

    Solo cargar artefactos joblib generados por nosotros: deserializar un fichero
    no confiable puede ejecutar código.
    """
    ruta_modelo = resolve_path(cfg["artefactos"]["modelo"])
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No existe {ruta_modelo}. Entrena antes un modelo con: "
            "uv run python -m mlops_course.train"
        )
    modelo = joblib.load(ruta_modelo)

    ruta_metadatos = resolve_path(cfg["artefactos"]["metadatos"])
    if not ruta_metadatos.exists():
        raise FileNotFoundError(
            f"No existe {ruta_metadatos}. Entrena antes un modelo con: "
            "uv run python -m mlops_course.train"
        )
    with ruta_metadatos.open("r", encoding="utf-8") as f:
        metadatos = json.load(f)

    return modelo, metadatos


def load_registered_model(cfg: dict, version: str):
    """Carga una versión concreta del Model Registry de MLflow."""
    mlflow.set_tracking_uri(tracking_uri(cfg))
    nombre = cfg["mlflow"]["modelo_registrado"]
    return mlflow.sklearn.load_model(f"models:/{nombre}/{version}")


def validar_columnas(df: pd.DataFrame) -> None:
    """Errores de estructura: impiden procesar el archivo entero."""
    if df.empty:
        raise ValueError("El CSV de entrada no tiene filas.")
    if TARGET in df.columns:
        raise ValueError(
            f"El CSV de entrada no debe incluir la columna '{TARGET}': "
            "es lo que el modelo predice."
        )
    faltan = [c for c in FEATURES if c not in df.columns]
    sobran = [c for c in df.columns if c not in FEATURES]
    if faltan or sobran:
        partes = []
        if faltan:
            partes.append(f"faltan {faltan}")
        if sobran:
            partes.append(f"no se reconocen {sobran}")
        raise ValueError(
            f"Columnas incorrectas: {'; '.join(partes)}. "
            f"Se esperan exactamente estas 13: {FEATURES}"
        )


def validar_filas(df: pd.DataFrame) -> pd.Series:
    """Errores de valores, fila a fila.

    Devuelve una Serie con el mismo índice que `df`: cadena vacía si la fila es
    válida, o la lista de problemas encontrados en esa fila.
    """
    errores = pd.Series("", index=df.index, dtype="object")

    def anotar(mascara: pd.Series, mensaje: str) -> None:
        errores[mascara] = errores[mascara] + mensaje + "; "

    for col in FEATURES:
        texto = df[col]
        valores = pd.to_numeric(texto, errors="coerce")  # texto no numérico -> NaN
        vacio = texto.isna()
        no_numerico = valores.isna() & ~vacio

        anotar(no_numerico, f"{col} no es un número")
        if col not in OPCIONALES:
            anotar(vacio, f"{col} es obligatorio y está vacío")

        presente = valores.notna()
        if col in NUMERICAS:
            minimo, maximo = RANGOS_NUMERICOS[col]
            fuera = presente & ~valores.between(minimo, maximo)
            anotar(fuera, f"{col} fuera de rango [{minimo}, {maximo}]")
        else:  # categórica: solo códigos del diccionario de datos
            admitidos = CATEGORIAS[col]
            fuera = presente & ~valores.isin(admitidos)
            anotar(fuera, f"{col} no admitido (valores válidos: {admitidos})")

    return errores.str.rstrip("; ")


def predict_df(model, df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Devuelve `probabilidad` de la clase 1 y `prediccion` (0/1) para cada fila.

    Asume filas ya validadas. Mantiene el índice de `df` para poder unir el
    resultado con la entrada sin perder la correspondencia.
    """
    X = df[FEATURES].apply(pd.to_numeric)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= umbral).astype(int)
    return pd.DataFrame({"probabilidad": y_proba, "prediccion": y_pred}, index=df.index)


def batch_predict(cfg: dict, entrada: Path, salida: Path) -> pd.DataFrame:
    """Scoring batch: lee un CSV, valida, predice y guarda el resultado.

    - Un error de estructura (archivo, columnas) detiene el proceso.
    - Un error en una fila no detiene el resto: esa fila sale sin predicción y
      con el motivo en la columna `error`.
    """
    modelo, metadatos = load_model(cfg)
    if not entrada.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada {entrada}")

    df = pd.read_csv(entrada)
    validar_columnas(df)
    df = df[FEATURES]  # mismo orden de columnas que en entrenamiento

    errores = validar_filas(df)
    validas = errores == ""

    umbral = cfg["modelo"]["umbral"]
    resultado = df.copy()
    resultado["probabilidad"] = float("nan")
    resultado["prediccion"] = pd.NA
    if validas.any():
        pred = predict_df(modelo, df[validas], umbral)
        resultado.loc[validas, "probabilidad"] = pred["probabilidad"].round(4)
        resultado.loc[validas, "prediccion"] = pred["prediccion"]
    resultado["prediccion"] = resultado["prediccion"].astype("Int64")
    resultado["error"] = errores
    resultado["umbral"] = umbral
    resultado["model_version"] = str(metadatos["model_version"])

    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida, index=False)

    n_err = int((~validas).sum())
    # Cada problema empieza por el nombre de su columna: "thal no admitido ..."
    campos = [
        parte.split(" ")[0]
        for texto in errores[~validas]
        for parte in texto.split("; ")
    ]
    registrar_inferencia(
        cfg,
        "batch",
        str(metadatos["model_version"]),
        len(df),
        n_err,
        "validacion_fila" if n_err else "",
        campos,
    )
    print(
        f"Modelo versión {metadatos['model_version']} | {len(df)} filas leídas, "
        f"{len(df) - n_err} predichas, {n_err} con error -> {salida}"
    )
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción batch sobre un CSV")
    parser.add_argument("--entorno", default=None)
    parser.add_argument("--input", default="data/ejemplos/pacientes_validos.csv")
    parser.add_argument("--output", default="reports/predicciones.csv")
    args = parser.parse_args()
    cfg = load_config(args.entorno)

    try:
        resultado = batch_predict(
            cfg, resolve_path(args.input), resolve_path(args.output)
        )
    except (FileNotFoundError, ValueError) as e:
        # Mensaje comprensible en lugar de una traza de Python
        tipo = (
            "archivo_no_encontrado"
            if isinstance(e, FileNotFoundError)
            else "estructura"
        )
        registrar_inferencia(cfg, "batch", None, 0, 1, tipo)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    columnas = ["probabilidad", "prediccion", "error", "model_version"]
    print(resultado[columnas].to_string())


if __name__ == "__main__":
    main()
