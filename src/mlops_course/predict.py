"""Carga del modelo y predicción (en memoria y batch sobre CSV).

Uso batch (se crea en la sesión 1 y se usa en la sesión 3), desde la raíz del proyecto:

    uv run python -m mlops_course.predict --input data/churn_nuevos.csv \
        --output reports/predicciones.csv
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlops_course.config import load_config, resolve_path, tracking_uri
from mlops_course.data import FEATURES, load_data


def load_model(cfg: dict) -> tuple[Any, dict]:
    """Carga el pipeline local y sus metadatos desde las rutas de la configuración.

    Solo cargar artefactos joblib generados por nosotros: deserializar un fichero
    no confiable puede ejecutar código.
    """
    # Comprobar que existe cfg["artefactos"]["modelo"] (FileNotFoundError
    # con un mensaje útil si no), cargarlo con joblib.load y leer el JSON de
    # cfg["artefactos"]["metadatos"]. Devolver (modelo, metadatos).

    # Cargamos la ruta del modelo y comprobamos si existe
    ruta_modelo = resolve_path(cfg["artefactos"]["modelo"])
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No existe {ruta_modelo}. Para ello tienes que entrenar antes un modelo."
        )

    # Deserializamos nuestro modelo
    modelo = joblib.load(ruta_modelo)

    # Leemos nuestros metadatos (JSON)
    ruta_metadatos = resolve_path(cfg["artefactos"]["metadatos"])

    if not ruta_metadatos.exists():
        raise FileNotFoundError(
            f"No existe {ruta_metadatos}. Para ello tienes que entrenar antes "
            "un modelo y generar sus metadatos."
        )

    with ruta_metadatos.open("r", encoding="utf-8") as f:
        metadatos = json.load(f)

    return modelo, metadatos


def load_registered_model(cfg: dict, version: str):
    """Carga una versión concreta del Model Registry de MLflow."""
    # TODO S2: set_tracking_uri(tracking_uri(cfg)) y devolver
    # mlflow.sklearn.load_model("models:/<modelo_registrado>/<version>").
    raise NotImplementedError("TODO S2: load_registered_model")


def predict_df(model, df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Devuelve `probabilidad` de abandono y clase `abandona` para cada fila."""
    # Usar model.predict_proba sobre df[FEATURES], quedarse con la
    # columna de la clase 1 y aplicar el umbral. Devolver un DataFrame con las
    # columnas "probabilidad" y "abandona" y el mismo índice que df.

    # Obtenemos la probabilidad y la clase predicha
    y_proba = model.predict_proba(df[FEATURES])[:, 1]
    y_pred = (y_proba >= umbral).astype(int)

    # Construimos un dataframe con ambas variables
    df_resultados = pd.DataFrame(
        {"probabilidad": y_proba, "abandona": y_pred}, index=df.index
    )

    return df_resultados


def batch_predict(cfg: dict, entrada: Path, salida: Path) -> pd.DataFrame:
    """Scoring batch: lee un CSV, predice y guarda el resultado con la versión."""
    # TODO S1: cargar modelo y metadatos (load_model), leer el CSV de entrada,
    # comprobar columnas, unir las predicciones (predict_df) al DataFrame original,
    # añadir columnas "umbral" y "model_version", guardar en `salida` y devolverlo.
    modelo, metadatos = load_model(cfg)
    df = load_data(entrada, es_entrenamiento=False)

    # Hacemos la predicción con el modelo
    umbral = cfg["modelo"]["umbral"]
    df_prediccion = predict_df(model=modelo, df=df, umbral=umbral)

    # Creamos el diccionario
    df_resultado = df.join(df_prediccion)
    df_resultado["umbral"] = umbral
    df_resultado["model_version"] = metadatos["model_version"]

    # Verificamos si salida existe
    salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(salida, index=False)

    print(f"Predicciones generadas para {len(df_resultado)} filas (abandona o no)")

    return df_resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción batch sobre un CSV")
    parser.add_argument("--entorno", default=None)
    parser.add_argument("--input", default="data/churn_nuevos.csv")
    parser.add_argument("--output", default="reports/predicciones.csv")
    args = parser.parse_args()
    cfg = load_config(args.entorno)
    df_resultado = batch_predict(
        cfg, resolve_path(args.input), resolve_path(args.output)
    )

    print(df_resultado)


if __name__ == "__main__":
    main()
