"""Carga de datasets desde data/raw."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_data(filename: str | None = None) -> pd.DataFrame:
    """
    Carga un dataset crudo desde data/raw.

    Args:
        filename: Nombre del archivo a cargar. Si es None, debe definirse la lógica del proyecto.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        NotImplementedError: La implementación concreta depende del dataset del proyecto.
    """
    raise NotImplementedError(
        "Implementa load_raw_data() según el formato de tus datos en data/raw."
    )


def load_csv(path: Path) -> pd.DataFrame:
    """Carga un archivo CSV desde la ruta indicada."""
    return pd.read_csv(path)
