"""Entrenamiento de modelos de machine learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: BaseEstimator | None = None,
    output_path: Path | None = None,
) -> BaseEstimator:
    """
    Entrena un modelo y opcionalmente lo persiste en models/.

    Args:
        X: Features de entrenamiento.
        y: Variable objetivo.
        model: Estimador de scikit-learn. Si es None, debe definirse uno por defecto.
        output_path: Ruta donde guardar el artefacto entrenado.

    Returns:
        Modelo entrenado.

    Raises:
        NotImplementedError: La implementación concreta depende del proyecto.
    """
    raise NotImplementedError(
        "Implementa train_model() con el estimador y pipeline de tu proyecto."
    )


def save_model(model: Any, path: Path) -> None:
    """Guarda un modelo entrenado en disco."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
