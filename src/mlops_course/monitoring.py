"""Monitorización de drift en las entradas del modelo de Heart Disease.

Dos pasos, desde la raíz del proyecto:

    uv run python -m mlops_course.monitoring simular              # escenario "cambio"
    uv run python -m mlops_course.monitoring simular --escenario estable
    uv run python -m mlops_course.monitoring informe

1. `simular` guarda:
   - la REFERENCIA: las features de entrenamiento (lo que el modelo "conoce");
   - un lote ACTUAL: una COPIA del conjunto de test (datos reservados), sin la
     columna `target`, como llegarían los pacientes en producción.
   Con `--escenario cambio` (por defecto) se introduce un cambio controlado en
   dos variables de esa copia; con `estable` la copia se deja intacta y sirve
   como control. Nunca se modifica el CSV original ni el test de evaluación.

2. `informe` aplica el modelo guardado a ambos conjuntos, compara con Evidently
   las 13 variables y la probabilidad predicha, y guarda:
   - `reports/drift_report.html`: informe interactivo de Evidently;
   - `reports/drift_resumen.md`: tabla por variable con método, valor, umbral
     y decisión, más medias de referencia y actual.

Método (Evidently, DataDriftPreset, valores por defecto para muestras pequeñas):
- numéricas: test de Kolmogórov-Smirnov; drift si p-valor < 0.05;
- categóricas: chi-cuadrado (o Z-test si son binarias); drift si p-valor < 0.05;
- conjunto: se señala drift si al menos el 50 % de las columnas lo presentan.

Límites: drift en las entradas NO equivale a peor rendimiento. Sin etiquetas
del lote actual no se puede medir el rendimiento, y una alerta no justifica
reentrenar automáticamente. Con ~60 filas los tests tienen poca potencia y,
al hacer 14 tests a la vez, es esperable algún falso positivo por azar.
"""

import argparse
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from mlops_course.config import load_config, resolve_path
from mlops_course.data import CATEGORICAS, FEATURES, NUMERICAS, load_data, split_data
from mlops_course.predict import load_model, predict_df
from mlops_course.schemas import RANGOS_NUMERICOS

# Cambio controlado del escenario "cambio": una población de más edad y con
# colesterol más alto (p. ej., otro centro o un programa de cribado distinto).
CAMBIO_EDAD_ANOS = 10
CAMBIO_COLESTEROL_FACTOR = 1.15


def _particiones(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, test) con la misma partición que usa el entrenamiento."""
    df = load_data(resolve_path(cfg["datos"]["ruta"]))
    train, _, test = split_data(
        df, cfg["datos"]["test_size"], cfg["datos"]["val_size"], cfg["semilla"]
    )
    return train, test


def simular_lote(test: pd.DataFrame, escenario: str) -> pd.DataFrame:
    """Crea el lote "actual" a partir de una COPIA de test, sin `target`.

    No se remuestrea para inflar el tamaño: repetir 61 filas hasta 1000 no
    aporta información y haría que los tests parecieran más seguros de lo que son.
    """
    lote = test[FEATURES].copy().reset_index(drop=True)
    if escenario == "cambio":
        lote["age"] = lote["age"] + CAMBIO_EDAD_ANOS
        lote["chol"] = (lote["chol"] * CAMBIO_COLESTEROL_FACTOR).round().astype(int)
        # Se mantienen dentro del contrato de entrada para que el lote sea válido
        for col in ("age", "chol"):
            minimo, maximo = RANGOS_NUMERICOS[col]
            lote[col] = lote[col].clip(minimo, maximo)
    elif escenario != "estable":
        raise ValueError(f"Escenario desconocido: {escenario}")
    return lote


def con_probabilidad(model, df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Añade la probabilidad predicha para vigilar también la salida del modelo."""
    return df.assign(probabilidad=predict_df(model, df, umbral)["probabilidad"])


def informe_drift(referencia: pd.DataFrame, actual: pd.DataFrame, salida: Path):
    """Compara referencia y actual con Evidently y guarda el HTML.

    Devuelve (resumen por columna, resultado global de Evidently).
    """
    definicion = DataDefinition(
        numerical_columns=[*NUMERICAS, "probabilidad"],
        categorical_columns=CATEGORICAS,
    )
    columnas = [*FEATURES, "probabilidad"]
    ref = Dataset.from_pandas(referencia[columnas], data_definition=definicion)
    act = Dataset.from_pandas(actual[columnas], data_definition=definicion)
    snapshot = Report([DataDriftPreset()]).run(current_data=act, reference_data=ref)
    salida.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(salida))

    resumen, global_ = {}, {}
    for metrica in snapshot.dict()["metrics"]:
        config = metrica["config"]
        nombre = metrica["metric_name"]
        if nombre.startswith("ValueDrift("):
            metodo, umbral, valor = (
                config["method"],
                config["threshold"],
                metrica["value"],
            )
            # p-valor pequeño o distancia grande son evidencia de drift
            drift = valor < umbral if "p_value" in metodo else valor > umbral
            resumen[config["column"]] = {
                "metodo": metodo,
                "valor": round(float(valor), 4),
                "umbral": umbral,
                "drift": bool(drift),
            }
        elif nombre.startswith("DriftedColumnsCount("):
            global_ = {
                "columnas_con_drift": int(metrica["value"]["count"]),
                "proporcion": round(float(metrica["value"]["share"]), 3),
                "umbral_proporcion": config.get("drift_share", 0.5),
            }
    return resumen, global_


def escribir_resumen(
    resumen: dict,
    global_: dict,
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    version: str,
    salida: Path,
) -> None:
    """Guarda un resumen en Markdown para el README y la demo."""
    lineas = [
        "# Resumen de drift",
        "",
        f"- Modelo: versión {version}",
        f"- Referencia: {len(referencia)} filas (entrenamiento)",
        f"- Actual: {len(actual)} filas (copia de test, sin etiquetas)",
        f"- Columnas con drift: {global_.get('columnas_con_drift')} de {len(resumen)}"
        f" ({global_.get('proporcion')}); drift del conjunto si >= "
        f"{global_.get('umbral_proporcion')}",
        "",
        "| Columna | Método | Valor | Umbral | Drift | Media ref. | Media actual |",
        "|---|---|---|---|---|---|---|",
    ]
    for col, r in resumen.items():
        lineas.append(
            f"| {col} | {r['metodo']} | {r['valor']} | {r['umbral']} | "
            f"{'**sí**' if r['drift'] else 'no'} | "
            f"{referencia[col].mean():.3f} | {actual[col].mean():.3f} |"
        )
    salida.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula un lote y analiza su drift.")
    parser.add_argument("accion", choices=["simular", "informe"])
    parser.add_argument(
        "--escenario",
        default="cambio",
        choices=["cambio", "estable"],
        help="cambio: +10 años y colesterol x1.15; estable: copia sin cambios.",
    )
    parser.add_argument("--entorno", default=None)
    args = parser.parse_args()

    cfg = load_config(args.entorno)
    mon = cfg["monitorizacion"]
    ruta_ref, ruta_act = resolve_path(mon["referencia"]), resolve_path(mon["actual"])

    if args.accion == "simular":
        train, test = _particiones(cfg)
        ruta_ref.parent.mkdir(parents=True, exist_ok=True)
        train[FEATURES].to_csv(ruta_ref, index=False)
        lote = simular_lote(test, args.escenario)
        lote.to_csv(ruta_act, index=False)
        print(f"Referencia (train): {len(train)} filas -> {ruta_ref}")
        print(f"Lote actual ({args.escenario}): {len(lote)} filas -> {ruta_act}")
        for col in ("age", "chol"):
            print(
                f"  media {col}: referencia {train[col].mean():.1f}"
                f" | actual {lote[col].mean():.1f}"
            )
        return

    model, metadatos = load_model(cfg)
    umbral = cfg["modelo"]["umbral"]
    referencia = con_probabilidad(model, pd.read_csv(ruta_ref), umbral)
    actual = con_probabilidad(model, pd.read_csv(ruta_act), umbral)
    ruta_html = resolve_path(mon["informe"])
    resumen, global_ = informe_drift(referencia, actual, ruta_html)
    ruta_md = ruta_html.with_name("drift_resumen.md")
    version = str(metadatos["model_version"])
    escribir_resumen(resumen, global_, referencia, actual, version, ruta_md)

    print(f"Modelo {version} · drift por columna (p-valor < umbral => drift):")
    for col, r in resumen.items():
        marca = "DRIFT" if r["drift"] else "ok"
        print(
            f"  {col:<13} {marca:<5} {r['valor']:.4f}"
            f" (umbral {r['umbral']}, {r['metodo']})"
        )
    print(
        f"Columnas con drift: {global_.get('columnas_con_drift')} de {len(resumen)}"
        f" | prob. media: ref {referencia.probabilidad.mean():.3f}"
        f" -> actual {actual.probabilidad.mean():.3f}"
    )
    print(f"Informe: {ruta_html}\nResumen: {ruta_md}")


if __name__ == "__main__":
    main()
