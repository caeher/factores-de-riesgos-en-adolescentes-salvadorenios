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
    MENTAL_HEALTH_CONSTRUCT,
    TARGET_IMC,
    TARGET_MENTAL_HEALTH,
    get_excluded_feature_columns,
    get_mental_health_target_col,
    load_config,
)

TaskType = Literal["regression", "classification"]

# Demografía
DEMOGRAPHIC_FEATURES = ["Q1", "Q2"]

# Alimentación, higiene y actividad física (sin peso/altura)
BEHAVIOR_FEATURES = [
    "QN6",
    "QN7",
    "QN8",
    "QN9",
    "QN10",
    "QN11",
    "QN12",
    "QN13",
    "QN14",
    "QN49",
    "QN50",
    "QN51",
    "QN52",
    "qnfrvgg",
    "qnpa7g",
    "qnpe5g",
]

# Alcohol y drogas
SUBSTANCE_FEATURES = ["QN34", "QN35", "QN36", "QN37", "QN38", "QN39", "QN40"]

# Violencia, bullying y lesiones (no salud mental)
VIOLENCE_FEATURES = ["QN15", "QN16", "QN17", "QN18", "QN19", "QN20", "QN21"]

# Factores de protección y supervisión parental
PROTECTIVE_FEATURES = ["QN53", "QN54", "QN55", "QN56", "QN57", "QN58"]

# Comportamiento sexual (riesgo adicional, no leakage)
SEXUAL_BEHAVIOR_FEATURES = ["QN44", "QN45", "QN46", "QN47", "QN48", "qnc1g"]

# Regresión IMC: comportamiento sin peso/altura ni derivadas de peso.
REGRESSION_FEATURE_COLUMNS = (
    DEMOGRAPHIC_FEATURES
    + BEHAVIOR_FEATURES
    + SUBSTANCE_FEATURES
    + VIOLENCE_FEATURES
    + PROTECTIVE_FEATURES
)

# Clasificación: factores de riesgo/protección sin constructo de salud mental.
CLASSIFICATION_FEATURE_COLUMNS = (
    DEMOGRAPHIC_FEATURES
    + BEHAVIOR_FEATURES
    + SUBSTANCE_FEATURES
    + VIOLENCE_FEATURES
    + PROTECTIVE_FEATURES
    + SEXUAL_BEHAVIOR_FEATURES
)


def _get_classification_exclusions() -> set[str]:
    """Columnas excluidas de features de clasificación (leakage + target activo)."""
    excluded = set(get_excluded_feature_columns())
    excluded.update(MENTAL_HEALTH_CONSTRUCT)
    target_col = get_mental_health_target_col(load_config())
    excluded.add(target_col)
    return excluded


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
        excluded = _get_classification_exclusions()

    candidates = (
        REGRESSION_FEATURE_COLUMNS
        if task == "regression"
        else CLASSIFICATION_FEATURE_COLUMNS
    )

    return [col for col in candidates if col in df.columns and col not in excluded]


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    """
    Construye el preprocesador: imputación + escalado / One-Hot.

    Justificación Q vs QN: las QN son recodificaciones binarias/ordinales
    estandarizadas por la OMS, evitando duplicar información con las Q originales
    y reduciendo dimensionalidad frente a One-Hot Encoding masivo.
    """
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
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
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
