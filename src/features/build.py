"""Construcción de features para modelado."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera features a partir de datos preprocesados.

    Args:
        df: DataFrame preprocesado.

    Returns:
        DataFrame con features listas para entrenamiento.

    Raises:
        NotImplementedError: La implementación concreta depende del proyecto.
    """
    raise NotImplementedError(
        "Implementa build_features() con las transformaciones de tu pipeline."
    )
