"""Módulo de modelos: entrenamiento e inferencia."""

from fras.models.predict import predict
from fras.models.train import train_model

__all__ = ["train_model", "predict"]
