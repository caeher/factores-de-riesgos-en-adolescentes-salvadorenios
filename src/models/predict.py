"""Inferencia con modelos entrenados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def predict(model_path: Path, X: pd.DataFrame) -> pd.Series:
    """
    Genera predicciones con un modelo entrenado.

    Args:
        model_path: Ruta al artefacto del modelo (.joblib o .pkl).
        X: Features para inferencia.

    Returns:
        Serie con las predicciones.

    Raises:
        NotImplementedError: La implementación concreta depende del proyecto.
    """
    raise NotImplementedError(
        "Implementa predict() para cargar el modelo y generar predicciones."
    )


def load_model(path: Path) -> Any:
    """Carga un modelo serializado desde disco."""
    import joblib

    return joblib.load(path)
