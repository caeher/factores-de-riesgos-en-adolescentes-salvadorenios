"""Módulo de modelos: entrenamiento e inferencia."""

from .predict import predict
from .train import train_model

__all__ = ["train_model", "predict"]
