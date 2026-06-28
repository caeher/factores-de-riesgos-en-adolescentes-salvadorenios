"""Preprocesamiento y limpieza de datos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    IMC_MAX,
    IMC_MIN,
    MENTAL_HEALTH_ALTERNATIVE_COL,
    TARGET_IMC,
    TARGET_MENTAL_HEALTH,
    get_mental_health_target_col,
    load_config,
)
from data.load import replace_missing_sentinel


def compute_bmi(weight_kg: pd.Series, height_m: pd.Series) -> pd.Series:
    """
    Calcula el Índice de Masa Corporal (IMC).

    IMC = peso (kg) / estatura (m)^2
    """
    height_sq = height_m.astype(float) ** 2
    bmi = weight_kg.astype(float) / height_sq
    bmi = bmi.replace([np.inf, -np.inf], np.nan)
    return bmi


def build_mental_health_target(
    df: pd.DataFrame,
    primary_col: str | None = None,
    fallback_col: str = MENTAL_HEALTH_ALTERNATIVE_COL,
) -> pd.Series:
    """
    Construye la variable binaria Riesgo_Salud_Mental.

    GSHS QN24: ¿Consideró suicidarse en los últimos 12 meses? (1=Sí, 2=No)
    Alternativa QN22: ¿Se sintió solo/a la mayor parte del tiempo? (1=Sí, 2=No)

    Returns:
        Serie binaria: 1 = riesgo, 0 = sin riesgo, NaN = sin respuesta válida.
    """
    if primary_col is None:
        primary_col = get_mental_health_target_col(load_config())

    source_col = primary_col if primary_col in df.columns else fallback_col
    if source_col not in df.columns:
        raise ValueError(
            f"No se encontró columna de salud mental: {primary_col} ni {fallback_col}"
        )

    raw = df[source_col]
    target = pd.Series(np.nan, index=df.index, dtype=float)
    target.loc[raw == 1] = 1.0
    target.loc[raw == 2] = 0.0
    target.name = TARGET_MENTAL_HEALTH
    return target


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones de limpieza y crea variables objetivo.

    - Reemplaza centinela de nulos (defensivo).
    - Calcula IMC a partir de Q5 (peso) y Q4 (estatura).
    - Crea Riesgo_Salud_Mental a partir de QN24 (ideación suicida) o QN22 (soledad).
    - Filtra IMC fuera de rango clínico razonable.

    Args:
        df: DataFrame crudo o intermedio.

    Returns:
        DataFrame preprocesado listo para feature engineering.
    """
    processed = replace_missing_sentinel(df.copy())

    if "Q4" not in processed.columns or "Q5" not in processed.columns:
        raise ValueError("El dataset debe contener Q4 (estatura) y Q5 (peso).")

    processed[TARGET_IMC] = compute_bmi(processed["Q5"], processed["Q4"])
    processed[TARGET_MENTAL_HEALTH] = build_mental_health_target(processed)

    # Filtrar IMC imposibles o extremos.
    invalid_imc = (processed[TARGET_IMC] < IMC_MIN) | (processed[TARGET_IMC] > IMC_MAX)
    processed.loc[invalid_imc, TARGET_IMC] = np.nan

    return processed
