"""Contrato de entrada/salida del servicio de predicción (Heart Disease).

Lo comparten la API (valida la entrada y la respuesta) y el batch (mismos rangos
numéricos). Solo depende de Pydantic: quien lo importe no necesita cargar
scikit-learn ni el modelo.

Los valores admitidos de las categóricas repiten los de `data.CATEGORIAS`
(no se importan para no arrastrar pandas/scikit-learn). Un test comprueba que
ambas definiciones coinciden, para que no se desincronicen.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Límites de plausibilidad (errores de captura o de unidades), no criterios
# clínicos. Fuente única para la API y para el batch (predict.py).
RANGOS_NUMERICOS: dict[str, tuple[float, float]] = {
    "age": (1, 120),  # años
    "trestbps": (50, 250),  # presión arterial en reposo (mm Hg)
    "chol": (50, 700),  # colesterol sérico (mg/dl)
    "thalach": (50, 250),  # frecuencia cardiaca máxima
    "oldpeak": (0, 10),  # depresión del ST inducida por ejercicio
}


def _rango(col: str, descripcion: str, **kwargs):
    minimo, maximo = RANGOS_NUMERICOS[col]
    return Field(ge=minimo, le=maximo, description=descripcion, **kwargs)


class Paciente(BaseModel):
    """Una fila de entrada: las 13 variables predictoras. Sin `target`."""

    model_config = ConfigDict(
        extra="forbid",  # rechaza campos desconocidos, incluido `target`
        json_schema_extra={
            "examples": [
                {
                    "age": 59,
                    "trestbps": 138,
                    "chol": 271,
                    "thalach": 182,
                    "oldpeak": 0.0,
                    "sex": 1,
                    "cp": 4,
                    "fbs": 0,
                    "restecg": 2,
                    "exang": 0,
                    "slope": 1,
                    "ca": 0,
                    "thal": 3,
                }
            ]
        },
    )

    # Numéricas
    age: int = _rango("age", "Edad en años")
    trestbps: int = _rango("trestbps", "Presión arterial en reposo (mm Hg)")
    chol: int = _rango("chol", "Colesterol sérico (mg/dl)")
    thalach: int = _rango("thalach", "Frecuencia cardiaca máxima alcanzada")
    oldpeak: float = _rango("oldpeak", "Depresión del ST", allow_inf_nan=False)

    # Categóricas (códigos del diccionario de datos)
    sex: Literal[0, 1] = Field(description="0 mujer, 1 hombre")
    cp: Literal[1, 2, 3, 4] = Field(description="Tipo de dolor torácico")
    fbs: Literal[0, 1] = Field(description="Glucemia en ayunas > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="Electrocardiograma en reposo")
    exang: Literal[0, 1] = Field(description="Angina inducida por ejercicio")
    slope: Literal[1, 2, 3] = Field(description="Pendiente del segmento ST")
    # Pueden faltar (como en el dataset): el pipeline las imputa
    ca: Literal[0, 1, 2, 3] | None = Field(
        default=None, description="Nº de vasos principales (0-3); opcional"
    )
    thal: Literal[3, 6, 7] | None = Field(
        default=None, description="3 normal, 6 defecto fijo, 7 reversible; opcional"
    )


class Prediccion(BaseModel):
    """Respuesta de POST /predict."""

    prediccion: Literal[0, 1] = Field(description="0 sin enfermedad, 1 con enfermedad")
    probabilidad: float = Field(ge=0, le=1, allow_inf_nan=False)
    umbral: float = Field(ge=0, le=1, allow_inf_nan=False)
    model_version: str = Field(min_length=1)
