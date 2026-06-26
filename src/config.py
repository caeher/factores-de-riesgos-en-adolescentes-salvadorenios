"""Configuración y rutas del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

__version__ = "0.1.0"
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ProjectPaths:
    """Rutas principales del proyecto."""

    root: Path
    data: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_external: Path
    models: Path
    reports: Path
    reports_figures: Path
    configs: Path
    notebooks: Path


def get_project_root() -> Path:
    """Devuelve la raíz del proyecto (directorio que contiene src/, data/, etc.)."""
    return Path(__file__).resolve().parents[1]


def get_project_paths() -> ProjectPaths:
    """Devuelve las rutas estándar del proyecto."""
    root = get_project_root()
    data = root / "data"
    reports = root / "reports"
    return ProjectPaths(
        root=root,
        data=data,
        data_raw=data / "raw",
        data_interim=data / "interim",
        data_processed=data / "processed",
        data_external=data / "external",
        models=root / "models",
        reports=reports,
        reports_figures=reports / "figures",
        configs=root / "configs",
        notebooks=root / "notebooks",
    )


def get_config_path() -> Path:
    """Devuelve la ruta al archivo de configuración activo."""
    paths = get_project_paths()
    local_config = paths.configs / "config.yaml"
    if local_config.exists():
        return local_config
    return paths.configs / "config.example.yaml"


def load_config() -> dict[str, Any]:
    """Carga la configuración YAML del proyecto."""
    load_dotenv()
    config_path = get_config_path()
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
