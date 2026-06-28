"""Construcción de features para modelado."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

from config import (
    TARGET_IMC,
    TARGET_MENTAL_HEALTH,
    get_excluded_feature_columns,
    get_qn_columns,
)

TaskType = Literal["regression", "classification"]

# Features de comportamiento para regresión IMC (sin peso/altura).
# QN de alimentación, actividad física, tiempo en pantalla y demografía básica.
REGRESSION_FEATURE_COLUMNS = [
    "Q1",  # edad
    "Q2",  # sexo
    "QN6",
    "QN7",
    "QN8",
    "QN9",
    "QN10",
    "QN11",
    "QN12", 
    "QN13",
    "QN14",
    "QN15",
    "QN34",
    "QN35",
    "QN36",
    "QN37",
    "QN38",
    "QN39",
    "QN40",
    "QN44",
    "QN45",
    "QN46",
    "QN47",
    "QN48",
    "QN49",
    "QN50",
    "QN51",
    "QN52",
    "qnfrvgg",
    "qnpa7g",
    "qnpe5g",
]

# Factores de protección/riesgo para clasificación de salud mental.
# Excluye variables del propio constructo de salud mental (QN21-QN27).
CLASSIFICATION_FEATURE_COLUMNS = [
    "Q1",
    "Q2",
    "QN6",
    "QN7",
    "QN8",
    "QN9",
    "QN10",
    "QN11",
    "QN12",
    "QN13",
    "QN14",
    "QN15",
    "QN16",
    "QN17",
    "QN18",
    "QN19",
    "QN20",
    "QN22",
    "QN23",
    "QN24",
    "QN28",
    "QN29",
    "QN30",
    "QN31",
    "QN32",
    "QN33",
    "QN34",
    "QN35",
    "QN36",
    "QN37",
    "QN38",
    "QN39",
    "QN40",
    "QN44",
    "QN45",
    "QN46",
    "QN47",
    "QN48",
    "QN49",
    "QN50",
    "QN51",
    "QN52",
    "QN53",
    "QN54",
    "QN55",
    "QN56",
    "QN57",
    "QN58",
    "qnc1g",
    "qnc2g",
]

# Variables de salud mental que no deben usarse como predictores.
MENTAL_HEALTH_LEAKAGE = [
    "QN21",
    "QN25",
    "QN26",
    "QN27",
    "Q21",
    "Q25",
    "Q26",
    "Q27",
]


def get_feature_columns(
    df: pd.DataFrame,
    task: TaskType,
) -> list[str]:
    """
    Selecciona columnas de features según la tarea, evitando data leakage.

    Usamos columnas QN (recodificaciones numéricas) en lugar de Q para evitar
    colinealidad entre respuestas originales y recodificadas.

    Args:
        df: DataFrame preprocesado.
        task: 'regression' para IMC o 'classification' para riesgo mental.

    Returns:    
        Lista de nombres de columnas disponibles en el DataFrame.
    """
    excluded = set(get_excluded_feature_columns())
    if task == "classification":
        excluded.update(MENTAL_HEALTH_LEAKAGE)

    if task == "regression":
        candidates = REGRESSION_FEATURE_COLUMNS
    else:
        candidates = CLASSIFICATION_FEATURE_COLUMNS

    available = [col for col in candidates if col in df.columns and col not in excluded]

    # Incluir QN adicionales presentes en el dataset que no estén excluidas.
    if task == "regression":
        extra_qn = [
            c
            for c in get_qn_columns()
            if c in df.columns and c not in excluded and c not in available
        ]
        available.extend(extra_qn[:5])  # límite conservador

    return available


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """
    Construye el preprocesador: imputación por mediana + RobustScaler.

    Justificación Q vs QN: las QN son recodificaciones binarias/ordinales
    estandarizadas por la OMS, evitando duplicar información con las Q originales
    y reduciendo dimensionalidad frente a One-Hot Encoding masivo.
    """

    #Identifica si Q1 o Q2 están en las columnas para tratarlas como categóricas
    categorical_cols = [c for c in ["Q1", "Q2"] if c in feature_columns]
    numeric_cols = [c for c in feature_columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), # Imputación por moda
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")), # Codificación
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols), # <-- Procesa Q1 y Q2 con One-Hot
        ],
        remainder="drop",
    )


def build_features(
    df: pd.DataFrame,
    task: TaskType,
) -> tuple[pd.DataFrame, list[str], ColumnTransformer]:
    """
    Prepara features para modelado.

    Args:
        df: DataFrame preprocesado.
        task: Tipo de tarea ML.

    Returns:
        Tupla (X, columnas_usadas, preprocesador).
    """
    feature_columns = get_feature_columns(df, task)
    if not feature_columns:
        raise ValueError(f"No hay columnas de features disponibles para {task}.")

    X = df[feature_columns].copy()
    preprocessor = build_preprocessor(feature_columns)
    return X, feature_columns, preprocessor


def get_target(
    df: pd.DataFrame,
    task: TaskType,
) -> pd.Series:
    """Devuelve la variable objetivo según la tarea."""
    if task == "regression":
        return df[TARGET_IMC]
    return df[TARGET_MENTAL_HEALTH]
