"""Preprocesamiento y limpieza de datos."""

from __future__ import annotations

import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones de limpieza y preprocesamiento.

    Args:
        df: DataFrame crudo o intermedio.

    Returns:
        DataFrame preprocesado listo para feature engineering.

    Raises:
        NotImplementedError: La implementación concreta depende del proyecto.
    """
    raise NotImplementedError(
        "Implementa preprocess_data() con las reglas de limpieza de tu dataset."
    )
