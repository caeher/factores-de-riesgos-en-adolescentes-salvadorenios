"""Gráficos y visualizaciones para análisis y reportes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import TARGET_IMC, TARGET_MENTAL_HEALTH


def set_plot_style() -> None:
    """Aplica un estilo consistente para todas las visualizaciones."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "font.size": 11,
        }
    )


def _save_figure(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    output_path: Path | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    bins: int = 30,
) -> plt.Figure:
    """
    Genera un histograma de la distribución de una columna numérica.

    Args:
        df: DataFrame con los datos.
        column: Nombre de la columna a visualizar.
        output_path: Si se indica, guarda la figura en reports/figures/.
        title: Título personalizado del gráfico.
        xlabel: Etiqueta del eje X.
        bins: Número de bins del histograma.

    Returns:
        Figura de matplotlib generada.
    """
    set_plot_style()
    fig, ax = plt.subplots()
    data = df[column].dropna()

    sns.histplot(data=data, bins=bins, kde=True, ax=ax, color="#2E86AB")
    ax.set_title(title or f"Distribución de {column}")
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel("Frecuencia")

    _save_figure(fig, output_path)
    return fig


def plot_age_distribution(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Distribución de edades de los adolescentes encuestados (Q1)."""
    age_labels = {
        1: "≤13",
        2: "14",
        3: "15",
        4: "16",
        5: "≥17",
    }
    counts = df["Q1"].map(age_labels).value_counts().reindex(age_labels.values())
    set_plot_style()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax, color="#A23B72")
    ax.set_title("Distribución de edades — adolescentes GSHS El Salvador 2013")
    ax.set_xlabel("Grupo de edad (años)")
    ax.set_ylabel("Número de estudiantes")
    ax.tick_params(axis="x", rotation=0)
    _save_figure(fig, output_path)
    return fig


def plot_bmi_distribution(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Distribución del IMC calculado."""
    return plot_distribution(
        df,
        TARGET_IMC,
        output_path=output_path,
        title="Distribución del Índice de Masa Corporal (IMC)",
        xlabel="IMC (kg/m²)",
    )


def plot_mental_health_prevalence(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Prevalencia del riesgo de salud mental en la muestra."""
    set_plot_style()
    counts = df[TARGET_MENTAL_HEALTH].value_counts().sort_index()
    labels = ["Sin riesgo (0)", "Riesgo (1)"]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#4CAF50", "#E53935"])
    ax.set_title("Prevalencia de riesgo en salud mental (ideación suicida seria)")
    ax.set_ylabel("Número de estudiantes")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )
    _save_figure(fig, output_path)
    return fig


def plot_confusion_matrix(
    confusion: np.ndarray,
    output_path: Path | None = None,
    title: str = "Matriz de confusión — Riesgo salud mental",
) -> plt.Figure:
    """Matriz de confusión detallada para clasificación."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred: Sin riesgo", "Pred: Riesgo"],
        yticklabels=["Real: Sin riesgo", "Real: Riesgo"],
        ax=ax,
    )
    ax.set_title(title)
    _save_figure(fig, output_path)
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    output_path: Path | None = None,
) -> plt.Figure:
    """Curva ROC con AUC anotado."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565C0", lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Azar")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curva ROC — AUC = {auc_score:.3f}")
    ax.legend(loc="lower right")
    _save_figure(fig, output_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path | None = None,
) -> plt.Figure:
    """Análisis de residuos para el modelo de regresión IMC."""
    residuals = y_true - y_pred
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#6A4C93")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("IMC predicho")
    axes[0].set_ylabel("Residuo")
    axes[0].set_title("Residuos vs. predicciones")

    sns.histplot(residuals, kde=True, ax=axes[1], color="#6A4C93")
    axes[1].set_title("Distribución de residuos")
    axes[1].set_xlabel("Residuo")

    fig.suptitle("Análisis de residuos — Regresión IMC", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_feature_importance(
    importances: dict[str, float],
    output_path: Path | None = None,
    title: str = "Importancia de características (Top 15)",
    top_n: int = 15,
) -> plt.Figure:
    """Gráfico de barras horizontales de feature importance."""
    set_plot_style()
    items = list(importances.items())[:top_n]
    names = [i[0] for i in reversed(items)]
    values = [i[1] for i in reversed(items)]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(names, values, color="#F18F01")
    ax.set_title(title)
    ax.set_xlabel("Importancia relativa")
    _save_figure(fig, output_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str],
    output_path: Path | None = None,
) -> plt.Figure:
    """Mapa de calor de correlaciones bivariadas."""
    set_plot_style()
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlaciones entre variables seleccionadas")
    _save_figure(fig, output_path)
    return fig
