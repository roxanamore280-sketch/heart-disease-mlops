"""API de predicción de enfermedad cardiaca con FastAPI.

Arranque (desde la raíz del proyecto):

    uv run uvicorn mlops_course.api:app --host 127.0.0.1 --port 8000

Documentación interactiva: http://127.0.0.1:8000/docs

Proyecto académico: no es una herramienta de diagnóstico.
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mlops_course.config import load_config
from mlops_course.data import TARGET
from mlops_course.predict import load_model, predict_df, registrar_inferencia
from mlops_course.schemas import Error, ErrorValidacion, Paciente, Prediccion, Salud

estado: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo UNA vez al arrancar, no en cada petición.

    Si el artefacto no existe, la API arranca igualmente pero responde 503 con
    un mensaje que explica cómo generarlo (en vez de caerse sin explicación).
    """
    cfg = load_config()
    try:
        modelo, metadatos = load_model(cfg)
        estado["modelo"] = modelo
        estado["umbral"] = cfg["modelo"]["umbral"]
        estado["model_version"] = str(metadatos["model_version"])
        estado["cfg"] = cfg
    except FileNotFoundError as e:
        estado["error_carga"] = str(e)
        estado["cfg"] = cfg
    yield
    estado.clear()


app = FastAPI(
    title="API de predicción de enfermedad cardiaca (Heart Disease)",
    version="1.0",
    lifespan=lifespan,
)


def _traducir(error: dict) -> dict:
    """Convierte un error de Pydantic en un mensaje comprensible en español."""
    campo = ".".join(str(p) for p in error["loc"] if p != "body") or "body"
    tipo = error["type"]
    ctx = error.get("ctx", {})
    if tipo == "missing":
        problema = "campo obligatorio ausente"
    elif tipo == "extra_forbidden" and campo == TARGET:
        problema = f"'{TARGET}' no puede enviarse: es lo que el modelo predice"
    elif tipo == "extra_forbidden":
        problema = "campo no reconocido"
    elif tipo == "literal_error":
        validos = ctx.get("expected", "").replace(" or ", ", ")
        problema = f"valor no admitido; valores válidos: {validos}"
    elif tipo == "greater_than_equal":
        problema = f"fuera de rango: debe ser >= {ctx.get('ge')}"
    elif tipo == "less_than_equal":
        problema = f"fuera de rango: debe ser <= {ctx.get('le')}"
    elif tipo in ("int_parsing", "int_from_float", "float_parsing"):
        problema = "debe ser un número"
    else:
        problema = error["msg"]
    recibido = None if tipo == "missing" else error.get("input")
    return {"campo": campo, "valor_recibido": recibido, "problema": problema}


@app.exception_handler(RequestValidationError)
async def errores_de_validacion(request: Request, exc: RequestValidationError):
    """422 con un formato legible: qué campo falla y por qué."""
    errores = [_traducir(e) for e in exc.errors()]
    if "cfg" in estado:
        registrar_inferencia(
            estado["cfg"],
            "api",
            estado.get("model_version"),
            1,
            1,
            "validacion",
            [e["campo"] for e in errores],
        )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "La entrada no cumple el contrato de la API",
            "errores": errores,
        },
    )


@app.get(
    "/health",
    response_model=Salud,
    responses={503: {"model": Salud, "description": "Modelo no cargado"}},
)
def health():
    """Comprobación de estado: si hay modelo cargado y qué versión."""
    if "modelo" not in estado:
        return JSONResponse(
            status_code=503,
            content={"status": "sin_modelo", "detail": estado.get("error_carga")},
        )
    return Salud(status="ok", model_version=estado["model_version"])


@app.post(
    "/predict",
    response_model=Prediccion,
    responses={
        422: {"model": ErrorValidacion, "description": "Entrada no válida"},
        503: {"model": Error, "description": "Modelo no disponible"},
    },
)
def predict(paciente: Paciente) -> Prediccion:
    """Predice la probabilidad de enfermedad cardiaca (clase 1) de un paciente.

    Si la entrada no cumple el contrato, responde 422 indicando el campo y el motivo.
    """
    if "modelo" not in estado:
        if "cfg" in estado:
            registrar_inferencia(estado["cfg"], "api", None, 1, 1, "sin_modelo")
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {estado.get('error_carga')}",
        )
    # ca/thal ausentes llegan como None -> NaN, y el pipeline los imputa
    df = pd.DataFrame([paciente.model_dump()])
    resultado = predict_df(estado["modelo"], df, estado["umbral"]).iloc[0]
    registrar_inferencia(estado["cfg"], "api", estado["model_version"], 1, 0)
    return Prediccion(
        prediccion=int(resultado["prediccion"]),
        probabilidad=round(float(resultado["probabilidad"]), 4),
        umbral=estado["umbral"],
        model_version=estado["model_version"],
    )
