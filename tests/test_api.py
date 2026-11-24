"""Comprobaciones automáticas de la inferencia: API, batch, contrato y registro.

Ejecución, desde la raíz del proyecto:

    uv run pytest tests/test_api.py -v

Los tests NO dependen de haber ejecutado `train` antes: un fixture entrena un
pipeline pequeño con el mismo `build_pipeline` del proyecto, lo guarda con
joblib en una carpeta temporal y apunta la configuración allí. Así se prueba
la carga real de un artefacto desde disco sin tocar `artifacts/` ni `logs/`.
"""

import json
import typing
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_course import api as api_module
from mlops_course.config import load_config, resolve_path
from mlops_course.data import CATEGORIAS, FEATURES, load_data, split_data, x_y
from mlops_course.models import build_pipeline
from mlops_course.predict import CAMPOS_REGISTRO, batch_predict, load_model
from mlops_course.schemas import Paciente

EJEMPLOS = resolve_path("data/ejemplos")
PACIENTE_VALIDO = json.loads((EJEMPLOS / "paciente_valido.json").read_text())


@pytest.fixture(scope="session")
def modelo_entrenado(tmp_path_factory) -> tuple[Path, Path]:
    """Entrena y guarda un pipeline en una carpeta temporal (una vez por sesión)."""
    cfg = load_config()
    carpeta = tmp_path_factory.mktemp("artefactos")
    df = load_data(resolve_path(cfg["datos"]["ruta"]))
    train, _, _ = split_data(df, 0.2, 0.2, cfg["semilla"])
    modelo = build_pipeline("regresion_logistica", {"C": 1.0, "max_iter": 1000})
    modelo.fit(*x_y(train))
    ruta_modelo, ruta_meta = carpeta / "model.joblib", carpeta / "metadata.json"
    joblib.dump(modelo, ruta_modelo)
    ruta_meta.write_text(json.dumps({"model_version": "test"}))
    return ruta_modelo, ruta_meta


@pytest.fixture
def cfg(modelo_entrenado, tmp_path) -> dict:
    """Configuración real, pero con artefacto y registro en carpetas temporales."""
    cfg = load_config()
    cfg["artefactos"]["modelo"] = str(modelo_entrenado[0])
    cfg["artefactos"]["metadatos"] = str(modelo_entrenado[1])
    cfg["api"]["inferencias"] = str(tmp_path / "inferencias.csv")
    return cfg


@pytest.fixture
def cliente(cfg, monkeypatch):
    """Cliente HTTP de la API con el modelo temporal ya cargado."""
    monkeypatch.setattr(api_module, "load_config", lambda: cfg)
    with TestClient(api_module.app) as c:
        yield c


# --- Carga del pipeline ------------------------------------------------------


def test_pipeline_se_carga_desde_disco_y_predice(cfg):
    modelo, metadatos = load_model(cfg)
    proba = modelo.predict_proba(pd.DataFrame([PACIENTE_VALIDO])[FEATURES])
    assert proba.shape == (1, 2)
    assert metadatos["model_version"] == "test"


def test_sin_artefacto_el_error_explica_como_generarlo(cfg, tmp_path):
    cfg["artefactos"]["modelo"] = str(tmp_path / "no_existe.joblib")
    with pytest.raises(FileNotFoundError, match="mlops_course.train"):
        load_model(cfg)


# --- API: casos válidos -------------------------------------------------------


def test_health_indica_modelo_y_version(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"] == "test"


def test_predict_valido_devuelve_clase_probabilidad_y_version(cliente):
    r = cliente.post("/predict", json=PACIENTE_VALIDO)
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["prediccion"] in (0, 1)
    assert 0 <= cuerpo["probabilidad"] <= 1
    assert cuerpo["model_version"] == "test"


def test_predict_acepta_ca_y_thal_ausentes(cliente):
    sin_opcionales = {
        k: v for k, v in PACIENTE_VALIDO.items() if k not in ("ca", "thal")
    }
    assert cliente.post("/predict", json=sin_opcionales).status_code == 200


# --- API: rechazo de entradas inválidas ---------------------------------------


@pytest.mark.parametrize(
    ("cambio", "campo"),
    [
        ({"thal": 5}, "thal"),  # código no admitido
        ({"age": -3}, "age"),  # fuera de rango
        ({"chol": "alto"}, "chol"),  # no numérico
        ({"target": 1}, "target"),  # el target nunca es una entrada
        ({"columna_inventada": 1}, "columna_inventada"),  # campo desconocido
    ],
)
def test_predict_rechaza_entrada_invalida_con_422(cliente, cambio, campo):
    r = cliente.post("/predict", json=PACIENTE_VALIDO | cambio)
    assert r.status_code == 422
    assert campo in [e["campo"] for e in r.json()["errores"]]


def test_predict_rechaza_campo_obligatorio_ausente(cliente):
    sin_sex = {k: v for k, v in PACIENTE_VALIDO.items() if k != "sex"}
    r = cliente.post("/predict", json=sin_sex)
    assert r.status_code == 422
    assert r.json()["errores"][0] == {
        "campo": "sex",
        "valor_recibido": None,
        "problema": "campo obligatorio ausente",
    }


def test_api_sin_modelo_responde_503(cfg, tmp_path, monkeypatch):
    cfg["artefactos"]["modelo"] = str(tmp_path / "no_existe.joblib")
    monkeypatch.setattr(api_module, "load_config", lambda: cfg)
    with TestClient(api_module.app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=PACIENTE_VALIDO).status_code == 503


# --- Batch ----------------------------------------------------------------------


def test_batch_conserva_filas_y_orden(cfg, tmp_path):
    entrada = EJEMPLOS / "pacientes_validos.csv"
    resultado = batch_predict(cfg, entrada, tmp_path / "pred.csv")
    original = pd.read_csv(entrada)
    assert len(resultado) == len(original)
    pd.testing.assert_series_equal(resultado["age"], original["age"])
    assert (resultado["error"] == "").all()
    assert resultado["prediccion"].isin([0, 1]).all()


def test_batch_marca_filas_invalidas_sin_detener_el_resto(cfg, tmp_path):
    resultado = batch_predict(
        cfg, EJEMPLOS / "pacientes_invalidos.csv", tmp_path / "pred.csv"
    )
    assert (resultado["error"] != "").sum() == 4
    assert resultado.loc[4, "prediccion"] in (0, 1)  # la fila válida sí se predice


def test_batch_rechaza_columnas_incorrectas(cfg, tmp_path):
    with pytest.raises(ValueError, match="faltan"):
        batch_predict(
            cfg, EJEMPLOS / "pacientes_sin_columna.csv", tmp_path / "pred.csv"
        )


def test_batch_y_api_dan_la_misma_probabilidad(cfg, cliente, tmp_path):
    """Las dos vías reutilizan el mismo pipeline guardado."""
    resultado = batch_predict(
        cfg, EJEMPLOS / "pacientes_validos.csv", tmp_path / "pred.csv"
    )
    r = cliente.post("/predict", json=PACIENTE_VALIDO)  # = fila 0 del CSV
    assert r.json()["probabilidad"] == pytest.approx(
        resultado.loc[0, "probabilidad"], abs=1e-4
    )


# --- Registro de inferencias ----------------------------------------------------


def test_registro_no_guarda_valores_clinicos(cfg, cliente):
    cliente.post("/predict", json=PACIENTE_VALIDO)
    cliente.post("/predict", json=PACIENTE_VALIDO | {"thal": 5})
    registro = pd.read_csv(cfg["api"]["inferencias"])
    assert list(registro.columns) == CAMPOS_REGISTRO
    assert list(registro["n_errores"]) == [0, 1]
    assert not set(FEATURES) & set(registro.columns)


# --- Contrato ---------------------------------------------------------------------


def test_schema_y_datos_admiten_los_mismos_codigos():
    """schemas.py no importa data.py: este test evita que se desincronicen."""
    for col, admitidos in CATEGORIAS.items():
        args = typing.get_args(Paciente.model_fields[col].annotation)
        if type(None) in args:  # ca y thal: Literal[...] | None
            args = typing.get_args(args[0])
        assert sorted(args) == sorted(admitidos), col
