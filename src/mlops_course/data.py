"""Carga, validación y partición de datos del caso de abandono (churn)."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

NUMERICAS = ["antiguedad_meses", "cuota_mensual"]
CATEGORICAS = ["tipo_contrato", "tipo_internet", "metodo_pago"]
FEATURES = NUMERICAS + CATEGORICAS
TARGET = "abandona"

CATEGORIAS = {
    "tipo_contrato": ["mensual", "anual", "bianual"],
    "tipo_internet": ["dsl", "fibra", "sin_internet"],
    "metodo_pago": ["cheque_electronico", "cheque_postal", "transferencia", "tarjeta"],
}


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

    # Comprobamos las columnas faltantes si es para entrenamiento o prediccion
    if es_entrenamiento:
        lista_columnas = [*FEATURES, TARGET]
    else:
        lista_columnas = [*FEATURES]

    columnas_faltantes = [c for c in lista_columnas if c not in df.columns]

    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas en el CSV cuya ruta es {ruta}: {columnas_faltantes}"
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
