"""Carga, validación y partición de datos del caso Heart Disease (UCI Cleveland)."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Variables según el diccionario de datos (data/alt/README.md)
NUMERICAS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERICAS + CATEGORICAS
TARGET = "target"

# Valores admitidos de cada variable categórica (códigos del dataset original).
# `ca` es un recuento de vasos (0-3); se trata como categórica, igual que en el
# notebook del profesor.
CATEGORIAS = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}


def calcular_hash(ruta: str | Path) -> str:
    """Devuelve el hash SHA-256 de un archivo (su "huella")."""
    contenido = Path(ruta).read_bytes()
    return hashlib.sha256(contenido).hexdigest()


def comprobar_hash(ruta_csv: str | Path, ruta_metadatos: str | Path) -> str:
    """Comprueba que el CSV es la copia común facilitada por el profesor.

    Compara el hash del CSV con el `csv_sha256` del archivo de metadatos.
    Devuelve el hash si coincide; si no, lanza ValueError.
    """
    with Path(ruta_metadatos).open(encoding="utf-8") as f:
        metadatos = json.load(f)

    hash_csv = calcular_hash(ruta_csv)
    if hash_csv != metadatos["csv_sha256"]:
        raise ValueError(
            f"El CSV {ruta_csv} no coincide con la copia común "
            f"({metadatos['version']}). ¿Se ha modificado el archivo?"
        )
    return hash_csv


def load_data(ruta: str | Path, es_entrenamiento: bool = True) -> pd.DataFrame:
    """Lee el CSV y comprueba que contiene las columnas esperadas.

    Args:
        ruta: Ruta al archivo CSV.
        es_entrenamiento: Indica si los datos son de entrenamiento
            (contendrán el target). Por defecto es True.

    Returns:
        DataFrame con los datos cargados.
    """
    df = pd.read_csv(ruta)

    # Comprobamos las columnas faltantes si es para entrenamiento o predicción
    if es_entrenamiento:
        lista_columnas = [*FEATURES, TARGET]
    else:
        lista_columnas = [*FEATURES]

    columnas_faltantes = [c for c in lista_columnas if c not in df.columns]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas en el CSV cuya ruta es {ruta}: {columnas_faltantes}"
        )

    # Para predecir, el target nunca debe venir en los datos de entrada
    if not es_entrenamiento and TARGET in df.columns:
        raise ValueError(
            f"El CSV de inferencia {ruta} no debe incluir la columna '{TARGET}'."
        )

    return df


def split_data(
    df: pd.DataFrame, test_size: float, val_size: float, semilla: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partición reproducible y estratificada en train / validación / test.

    `val_size` es la fracción de lo que queda tras separar test.
    Args:
        df: DataFrame con los datos completos.
        test_size: Fracción de datos a reservar para el conjunto de test.
        val_size: Fracción de datos a reservar para el conjunto de validación.
        semilla: Semilla para la aleatoriedad.

    Returns:
        Tupla con (train, validación, test) DataFrames.
    """
    df_resto, df_test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=semilla
    )

    df_train, df_val = train_test_split(
        df_resto, test_size=val_size, stratify=df_resto[TARGET], random_state=semilla
    )

    return df_train, df_val, df_test


def x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features y target."""
    X = df[FEATURES]
    y = df[TARGET]
    return X, y
