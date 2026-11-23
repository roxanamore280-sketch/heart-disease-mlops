"""Ejemplo didáctico de monitorización y actualización de un modelo de churn.

El módulo permite recorrer tres pasos:

1. `simular` crea una referencia con los datos de entrenamiento y una ventana
   actual de tamaño fijo. Se puede elegir entre datos estables, drift de
   covariables o un cambio controlado en la relación entre features y target.
2. `informe` aplica el modelo actual a ambas ventanas, compara sus features y
   probabilidades predichas con Evidently y guarda un informe HTML.
3. `comparar` entrena un candidato con parte de las etiquetas de la ventana
   actual y compara ambos modelos sobre la parte restante.

Ejecución (desde la raíz del proyecto):

    uv run python -m mlops_course.monitoring simular --escenario estable
    uv run python -m mlops_course.monitoring simular --escenario covariables
    uv run python -m mlops_course.monitoring simular --escenario concepto
    uv run python -m mlops_course.monitoring informe
    uv run python -m mlops_course.monitoring comparar

Entrena el modelo antes de ejecutar `informe` o `comparar`; `simular` no
necesita que exista un modelo entrenado.

Las acciones se ejecutan por separado y comparten los CSV definidos en la
configuración de monitorización. Por eso, ejecuta primero `simular`; después
puedes generar el informe y comparar modelos. `--escenario` solo afecta a esa
simulación y, si se omite, se usa `covariables`.

Interpretación de los escenarios:

- `estable`: remuestrea filas de test; sirve como control sin un cambio
  introducido intencionadamente.
- `covariables`: da más peso a cuotas altas, contratos mensuales y pago por
  cheque electrónico. Cambia la distribución de las features (P(X)), pero no
  modifica individualmente la etiqueta de las filas seleccionadas.
- `concepto`: conserva las features de una muestra como la estable y cambia a
  abandono algunas etiquetas de clientes bianuales. Cambia P(Y|X), aunque un
  detector que solo compara features puede no señalarlo.

Un resultado sin drift en las features no demuestra que el comportamiento del
target siga igual. Además, estos escenarios son simulaciones reproducibles para
practicar el flujo: no representan una fuente de datos en producción.
"""

import argparse

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from mlops_course.config import load_config, resolve_path
from mlops_course.data import (
    CATEGORICAS,
    FEATURES,
    NUMERICAS,
    TARGET,
    load_data,
    split_data,
    x_y,
)
from mlops_course.models import build_pipeline
from mlops_course.predict import load_model, predict_df
from mlops_course.train import evaluate

TAMANO_VENTANA = 1000
MARGEN_PROMOCION = 0.01  # mejora mínima de ROC AUC para promover un candidato


def _particiones(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los datos y devuelve sus particiones reproducibles.

    Args:
        cfg: Configuración con la ruta de datos, los tamaños de las particiones
            y la semilla aleatoria.

    Returns:
        Tupla `(train, val, test)`. En este módulo la referencia se construye
        con `train` y la ventana simulada con `test`.
    """
    df = load_data(resolve_path(cfg["datos"]["ruta"]))
    return split_data(
        df, cfg["datos"]["test_size"], cfg["datos"]["val_size"], cfg["semilla"]
    )


def simular_ventana(test: pd.DataFrame, escenario: str, semilla: int) -> pd.DataFrame:
    """Construye una ventana actual de tamaño fijo a partir de `test`.

    El remuestreo con reemplazo permite crear una ventana del tamaño configurado
    aunque `test` tenga menos filas. La semilla fija hace reproducible tanto la
    selección de filas como el cambio de etiquetas del escenario `concepto`.

    Args:
        test: Partición de test con features y target.
        escenario: `estable`, `covariables` o `concepto`.
        semilla: Semilla para las operaciones aleatorias.

    Returns:
        Copia remuestreada con índice consecutivo y, en el escenario `concepto`,
        posiblemente algunas etiquetas modificadas.

    Raises:
        ValueError: Si `escenario` no es uno de los valores admitidos.
    """
    rng = np.random.default_rng(semilla)
    if escenario in ("estable", "concepto"):
        ventana = test.sample(TAMANO_VENTANA, replace=True, random_state=semilla)
    elif escenario == "covariables":
        # Los pesos multiplicativos hacen más probable seleccionar filas que
        # cumplan varios criterios; no se alteran sus features ni sus etiquetas.
        pesos = (
            (test["cuota_mensual"] / test["cuota_mensual"].mean()) ** 2
            * np.where(test["tipo_contrato"] == "mensual", 2.0, 1.0)
            * np.where(test["metodo_pago"] == "cheque_electronico", 2.0, 1.0)
        )
        ventana = test.sample(
            TAMANO_VENTANA, replace=True, weights=pesos, random_state=semilla
        )
    else:
        raise ValueError(f"Escenario desconocido: {escenario}")

    ventana = ventana.reset_index(drop=True)
    if escenario == "concepto":
        # Simula que cambia P(Y|X) para parte del segmento bianual, sin cambiar X.
        segmento = ventana["tipo_contrato"] == "bianual"
        cambia = segmento & (rng.random(len(ventana)) < 0.4)
        ventana.loc[cambia, TARGET] = 1
    return ventana


def con_predicciones(model, df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    """Añade al DataFrame la probabilidad de abandono predicha por el modelo.

    Vigilar esta columna permite observar cambios en las salidas del modelo
    (prediction drift), además de comparar las distribuciones de entrada.

    Args:
        model: Modelo ya entrenado, compatible con `predict_df`.
        df: Datos que contienen las features requeridas por el modelo.
        umbral: Umbral de decisión configurado para la predicción.

    Returns:
        Copia de `df` con la columna `probabilidad`.
    """
    return df.assign(probabilidad=predict_df(model, df, umbral)["probabilidad"])


def informe_drift(referencia: pd.DataFrame, actual: pd.DataFrame, salida) -> dict:
    """Compara referencia y ventana actual y guarda el informe de Evidently.

    Se analizan las features y la probabilidad predicha, no el target. En el
    resumen, un p-valor bajo su umbral indica drift; para las métricas de
    distancia, un valor por encima del umbral indica drift.

    Args:
        referencia: Datos de referencia con features y `probabilidad`.
        actual: Datos actuales con las mismas columnas.
        salida: Ruta de destino del informe HTML.

    Returns:
        Diccionario indexado por columna. Cada valor contiene el método, el
        resultado redondeado, el umbral aplicado y el indicador booleano
        `drift`.
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
    print(f"Informe guardado en {salida}")

    resumen = {}
    for metrica in snapshot.dict()["metrics"]:
        config = metrica["config"]
        if metrica["metric_name"].startswith("ValueDrift("):
            metodo, umbral, valor = (
                config["method"],
                config["threshold"],
                metrica["value"],
            )
            # El sentido de la comparación depende del tipo de estadístico:
            # p-valor pequeño o distancia grande son evidencia de drift.
            drift = valor < umbral if "p_value" in metodo else valor > umbral
            resumen[config["column"]] = {
                "metodo": metodo,
                "valor": round(float(valor), 4),
                "umbral": umbral,
                "drift": bool(drift),
            }
    return resumen


def comparar_modelos(cfg: dict, train: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Evalúa un candidato frente al modelo actual con una ventana etiquetada.

    Una mitad de la ventana actual se trata como si sus etiquetas ya estuvieran
    disponibles y se añade a `train` para ajustar el candidato. La mitad
    restante se reserva para comparar ambos modelos sobre las mismas filas.
    Este ejemplo presupone que el target de `actual` está disponible; en un
    sistema real habría que esperar a que maduren las etiquetas.

    La decisión de promoción es solo una regla didáctica: el candidato debe
    mejorar ROC AUC al menos `MARGEN_PROMOCION` y no reducir F1. No despliega ni
    guarda automáticamente el candidato.

    Args:
        cfg: Configuración de modelo, umbral y semilla.
        train: Datos de entrenamiento originales.
        actual: Ventana actual con features y etiquetas.

    Returns:
        Diccionario con la versión del modelo actual, métricas de ambos modelos,
        la diferencia de ROC AUC y la decisión propuesta.
    """
    umbral = cfg["modelo"]["umbral"]
    actual_model, metadatos = load_model(cfg)
    maduras = actual.sample(frac=0.5, random_state=cfg["semilla"])
    evaluacion = actual.drop(maduras.index)

    # El pipeline incluye el preprocesamiento, que se ajusta junto con el
    # clasificador usando solo train más las filas maduras.
    candidato = build_pipeline(cfg["modelo"]["tipo"], cfg["modelo"]["parametros"])
    candidato.fit(*x_y(pd.concat([train, maduras])))

    # Los dos modelos se miden en la misma partición y con el mismo umbral.
    m_actual = evaluate(actual_model, *x_y(evaluacion), umbral)
    m_candidato = evaluate(candidato, *x_y(evaluacion), umbral)
    mejora = m_candidato["roc_auc"] - m_actual["roc_auc"]
    promover = mejora >= MARGEN_PROMOCION and m_candidato["f1"] >= m_actual["f1"]
    return {
        "version_actual": metadatos["model_version"],
        "actual": m_actual,
        "candidato": m_candidato,
        "mejora_roc_auc": mejora,
        "decision": "promover candidato" if promover else "mantener modelo actual",
    }


def main() -> None:
    """Interpreta los argumentos y ejecuta la acción de monitorización elegida."""
    parser = argparse.ArgumentParser(
        description="Simula drift, genera un informe o compara modelos."
    )
    parser.add_argument(
        "accion",
        choices=["simular", "informe", "comparar"],
        help="Paso del flujo: generar datos, analizar drift o evaluar un candidato.",
    )
    parser.add_argument(
        "--escenario",
        default="covariables",
        choices=["estable", "covariables", "concepto"],
        help="Cambio que se simula; solo se utiliza con la acción 'simular'.",
    )
    parser.add_argument(
        "--entorno", default=None, help="Entorno de configuración (p. ej., dev)."
    )
    args = parser.parse_args()

    cfg = load_config(args.entorno)
    mon = cfg["monitorizacion"]
    ruta_ref, ruta_act = resolve_path(mon["referencia"]), resolve_path(mon["actual"])
    train, _, test = _particiones(cfg)

    if args.accion == "simular":
        # La referencia es train; la ventana actual se deriva de test para
        # mantener separados los datos usados en cada distribución simulada.
        ruta_ref.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(ruta_ref, index=False)
        ventana = simular_ventana(test, args.escenario, cfg["semilla"])
        ventana.to_csv(ruta_act, index=False)
        print(f"Referencia (train): {len(train)} filas -> {ruta_ref}")
        print(f"Ventana actual ({args.escenario}): {len(ventana)} filas -> {ruta_act}")
        print(
            f"Tasa de abandono observada: ref {train[TARGET].mean():.3f}"
            f" | actual {ventana[TARGET].mean():.3f}"
        )

    elif args.accion == "informe":
        # Se calcula la salida del mismo modelo sobre ambos conjuntos para que
        # también se pueda comparar la distribución de sus probabilidades.
        model, metadatos = load_model(cfg)
        umbral = cfg["modelo"]["umbral"]
        referencia = con_predicciones(model, pd.read_csv(ruta_ref), umbral)
        actual = con_predicciones(model, pd.read_csv(ruta_act), umbral)
        resumen = informe_drift(referencia, actual, resolve_path(mon["informe"]))
        print(f"Modelo {metadatos['model_version']} · drift por columna:")
        for columna, r in resumen.items():
            marca = "DRIFT" if r["drift"] else "ok"
            print(
                f"  {columna:<18} {marca:<5} {r['valor']:.4f}"
                f" (umbral {r['umbral']}, {r['metodo']})"
            )
        print(
            f"Media de probabilidad predicha: ref {referencia.probabilidad.mean():.3f}"
            f" | actual {actual.probabilidad.mean():.3f}"
        )

    else:
        r = comparar_modelos(cfg, train, pd.read_csv(ruta_act))
        print(
            f"Modelo actual (versión {r['version_actual']}) vs candidato, mismos datos:"
        )
        for nombre in ("roc_auc", "f1", "recall", "precision"):
            print(
                f"  {nombre}: actual {r['actual'][nombre]:.4f}"
                f" | candidato {r['candidato'][nombre]:.4f}"
            )
        print(f"Decisión propuesta: {r['decision']}")
        print(f"Mejora de ROC AUC del candidato: {r['mejora_roc_auc']:+.4f}")


if __name__ == "__main__":
    main()
