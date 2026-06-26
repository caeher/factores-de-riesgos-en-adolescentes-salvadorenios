"""Gráficos y visualizaciones para análisis y reportes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Genera un histograma de la distribución de una columna.

    Args:
        df: DataFrame con los datos.
        column: Nombre de la columna a visualizar.
        output_path: Si se indica, guarda la figura en reports/figures/.

    Returns:
        Figura de matplotlib generada.

    Raises:
        NotImplementedError: La implementación concreta depende del proyecto.
    """
    raise NotImplementedError(
        "Implementa plot_distribution() con el estilo visual de tu proyecto."
    )


def set_plot_style() -> None:
    """Aplica un estilo consistente para todas las visualizaciones."""
    sns.set_theme(style="whitegrid")
