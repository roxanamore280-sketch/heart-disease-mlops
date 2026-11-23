"""Entrenamiento del modelo de abandono.

Uso (desde la raíz del proyecto):

    uv run python -m mlops_course.train               # baseline
    uv run python -m mlops_course.train --busqueda    # sesión 2: GridSearchCV acotado
    APP_ENV=preprod uv run python -m mlops_course.train
"""

import argparse
import json
from datetime import datetime
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from mlops_course.config import load_config, resolve_path, tracking_uri
from mlops_course.data import load_data, split_data, x_y
from mlops_course.models import build_pipeline


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, umbral: float) -> dict:
    """Métricas de clasificación con el umbral de decisión configurado."""
    # Calcular la probabilidad de la clase 1 con predict_proba, aplicar el
    # umbral y devolver un dict con roc_auc, f1, precision, recall y accuracy.
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)

    dict_metricas = {
        "roc_auc": roc_auc_score(y, y_prob),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
    }

    return dict_metricas


def save_artifacts(model: Pipeline, metadatos: dict[str, Any], cfg: dict) -> None:
    """Guarda el pipeline (joblib) y sus metadatos (JSON) en `artifacts/`."""
    # Crear la carpeta si no existe, guardar el modelo con joblib.dump en
    # cfg["artefactos"]["modelo"] y los metadatos con json.dump en
    # cfg["artefactos"]["metadatos"]. Usa resolve_path para las rutas.
    # Serializamos nuestro modelo
    ruta_modelo = resolve_path(cfg["artefactos"]["modelo"])
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filename=ruta_modelo)

    # Guardamos nuestros metadatos
    ruta_metadatos = resolve_path(cfg["artefactos"]["metadatos"])

    with ruta_metadatos.open("w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=2, ensure_ascii=False)

    print(f"Modelo guardado en {ruta_modelo} y metadatos en {ruta_metadatos}")


def search(cfg: dict, train: pd.DataFrame) -> tuple[Pipeline, dict, float]:
    """Búsqueda acotada de hiperparámetros con validación cruzada SOLO sobre train."""
    # TODO S2: crear un GridSearchCV con build_pipeline(...), param_grid,
    # cv=StratifiedKFold(..., shuffle=True, random_state=semilla) y scoring
    # tomados de cfg["busqueda"]. Ajustarlo SOLO con train y devolver
    # (best_estimator_, best_params_, best_score_).

    # Obtenemos las variables independientes y el target del conjunto de datos
    # de entrenamiento
    X_train, y_train = x_y(train)

    # Cargamos la configuración de búsqueda de hiperparámetros
    busqueda = cfg["busqueda"]

    # Creamos el pipeline
    modelo = build_pipeline(cfg["modelo"]["tipo"], cfg["modelo"]["parametros"])

    # Creamos nuestro grid search
    grid_search = GridSearchCV(
        estimator=modelo,
        param_grid=busqueda["param_grid"],
        scoring=busqueda["scoring"],
        # cv=busqueda["cv"],
        cv=StratifiedKFold(
            n_splits=busqueda["cv"], shuffle=True, random_state=cfg["semilla"]
        ),
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    # Mostramos los resultados y los devolvemos
    print(f"Mejor resultado de {busqueda['scoring']}: {grid_search.best_score_:.4f}")
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


def log_to_mlflow(
    cfg: dict,
    model: Pipeline,
    params: dict,
    metricas: dict[str, float],
    X_ejemplo: pd.DataFrame,
    run_name: str,
) -> tuple[str, str]:
    """Registra parámetros, métricas y modelo. Devuelve (run_id, versión registrada)."""
    # TODO S2: set_tracking_uri(tracking_uri(cfg)), set_experiment, y dentro de
    # mlflow.start_run(run_name=...): log_params, log_param("umbral", ...),
    # log_metrics, log_dict(cfg, "config.yaml") y mlflow.sklearn.log_model(...,
    # name="model", input_example=X_ejemplo, registered_model_name=...,
    # serialization_format="cloudpickle"). Devolver (run_id, versión registrada).

    # Indicamos cuál es la BBDD a utilizar de MLflow
    mlflow.set_tracking_uri(tracking_uri(cfg))

    # Seteamos el experimento (proyecto o caso de uso; por ejemplo, iris o churn)
    # sobre el que vamos a trabajar
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    # Hacemos log del modelo que acabamos de entrenar
    with mlflow.start_run(run_name=run_name) as run:
        # Hiperparámetros y otros valores de configuración
        mlflow.log_params(params)
        (mlflow.log_param("umbral", cfg["modelo"]["umbral"]),)

        # Guardamos las métricas
        mlflow.log_metrics(metricas)

        # Guardamos nuestro fichero de configuración
        mlflow.log_dict(cfg, "config.yaml")

        # Guardamos el modelo entrenado en MLflow
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_ejemplo,
            registered_model_name=cfg["mlflow"]["modelo_registrado"],
            serialization_format="cloudpickle",
        )

        # Guardamos la estructura del dataset (aparecerá en el apartado Dataset)
        dataset = mlflow.data.from_pandas(X_ejemplo)
        mlflow.log_input(dataset)

        # Guardamos una foto de los datos tanto en Parquet como CSV
        # Se verán en "Artifacts"
        mlflow.log_table(X_ejemplo, "datasets/muestra_churn.parquet")
        mlflow.log_artifact("./data/churn.csv", "datasets/churn.csv")

        # Ver en formato CSV
        # mlflow.log_input()
        # Etiquetamos la ejecución
        mlflow.set_tags({"entorno": cfg["entorno"], "tipo": cfg["modelo"]["tipo"]})

    # Obtenemos el run_id y la versión registrada
    run_id = run.info.run_id
    version = model_info.registered_model_version

    return run_id, version


def main() -> None:

    # Generamos los diferentes argumentos que recibirá nuestro script
    print("hola clase!! :)")
    parser = argparse.ArgumentParser(description="Entrena el modelo de abandono")
    parser.add_argument("--entorno", default=None, help="dev | preprod (o APP_ENV)")
    parser.add_argument("--busqueda", action="store_true", help="GridSearchCV acotado")
    args = parser.parse_args()

    # Cargar la configuración
    cfg = load_config(args.entorno)
    print(f"El parametro es: {args.busqueda}")

    # Cargar los datos y dividir en train, validación y test
    df = load_data(resolve_path(cfg["datos"]["ruta"]), es_entrenamiento=True)
    train, val, test = split_data(
        df,
        test_size=cfg["datos"]["test_size"],
        val_size=cfg["datos"]["val_size"],
        semilla=cfg["semilla"],
    )
    print(f"Filas -> train: {len(train)}, validación: {len(val)}, test: {len(test)}")
    X_train, y_train = x_y(train)
    umbral = cfg["modelo"]["umbral"]

    if args.busqueda:
        model, params, cv_score = search(cfg, train)
        run_name = "busqueda-grid"
    else:
        model = build_pipeline(cfg["modelo"]["tipo"], cfg["modelo"]["parametros"])
        model.fit(X_train, y_train)
        params = cfg["modelo"]["parametros"]
        run_name = "prueba_3"
        cv_score = None

    metricas = {f"val_{k}": v for k, v in evaluate(model, *x_y(val), umbral).items()}
    # El test se mira una vez, con el modelo ya elegido; no se usa para seleccionar.
    metricas |= {f"test_{k}": v for k, v in evaluate(model, *x_y(test), umbral).items()}
    if cv_score is not None:
        metricas["cv_mejor_score"] = cv_score
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")

    # TODO S2: sustituir esta línea por la llamada a log_to_mlflow(...)
    # datetime.now().strftime("local-%Y%m%d-%H%M%S")
    run_id, version = log_to_mlflow(
        cfg,
        model,
        params,
        metricas,
        X_train.head(5),
        run_name,
    )

    metadatos = {
        "model_version": version,
        "mlflow_run_id": run_id,
        "run_name": run_name,
        "tipo": cfg["modelo"]["tipo"],
        "parametros": params,
        "umbral": umbral,
        "entorno": cfg["entorno"],
        "entrenado_en": datetime.now().isoformat(timespec="seconds"),
        "metricas": metricas,
    }
    save_artifacts(model, metadatos, cfg)

    # TODO S2: recargar el artefacto con joblib.load y comprobar con np.allclose
    # que predict_proba sobre validación coincide con el modelo en memoria.

    # Recuperamos el modelo de aquellos que he registrado
    modelo_recuperado = mlflow.sklearn.load_model(
        f"models:/{cfg['mlflow']['modelo_registrado']}/{version}"
    )

    # Comprobamos que si hacemos predict_proba sobre la validación,
    # obtenemos los mismos resultados que con el modelo en memoria
    X_val, _ = x_y(val)
    assert np.allclose(
        modelo_recuperado.predict_proba(X_val), model.predict_proba(X_val)
    )

    df_resultado_val = modelo_recuperado.predict(X_val)
    print(df_resultado_val)
    print(f"Modelo del run {run_id} recuperado y verificado.")


if __name__ == "__main__":
    main()
