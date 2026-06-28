"""Configuración y rutas del proyecto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

__version__ = "0.1.0"
DEFAULT_RANDOM_SEED = 42

# Valor centinela de SPSS/Stata para datos faltantes en encuestas GSHS.
MISSING_SENTINEL = 1.79769313486232e308

# Archivo principal del desafío.
DEFAULT_RAW_FILENAME = "SLV2013_Public_Use.csv"

# Variables objetivo.
TARGET_IMC = "IMC"
TARGET_MENTAL_HEALTH = "Riesgo_Salud_Mental"

# Columnas con información directa de peso/altura (data leakage para regresión IMC).
LEAKAGE_COLUMNS = [
    "Q4",  # estatura (m)
    "Q5",  # peso (kg)
    "qnowtg",
    "qnobeseg",
    "qnunwtg",
    "weight",
]

# Variables de diseño muestral (no predictores).
SURVEY_DESIGN_COLUMNS = ["stratum", "psu"]

# Targets de salud mental (GSHS: 1=Sí, 2=No en recodificaciones QN).
# QN24: ideación suicida seriamente considerada (enlace V77 del desafío / fact sheet OMS).
MENTAL_HEALTH_PRIMARY_COL = "QN24"
# QN22: soledad la mayor parte del tiempo (escenario alternativo documentado).
MENTAL_HEALTH_ALTERNATIVE_COL = "QN22"
# Alias legacy para compatibilidad con tests.
MENTAL_HEALTH_FALLBACK_COL = MENTAL_HEALTH_ALTERNATIVE_COL

# Constructo completo de salud mental — nunca usar como features de clasificación.
MENTAL_HEALTH_CONSTRUCT = [
    "QN22",
    "QN23",
    "QN24",
    "QN25",
    "QN26",
    "QN27",
    "Q22",
    "Q23",
    "Q24",
    "Q25",
    "Q26",
    "Q27",
    "qnc2g",  # derivada de preocupación/bullying
]

# Mapeo de claves de configuración a columnas del dataset.
MENTAL_HEALTH_TARGET_MAP: dict[str, str] = {
    "qn24": "QN24",
    "qn22": "QN22",
    "qn25": "QN25",
}
DEFAULT_MENTAL_HEALTH_TARGET_KEY = "qn24"

# Rango clínico razonable de IMC para adolescentes (filtrado de outliers).
IMC_MIN = 10.0
IMC_MAX = 50.0


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


def get_random_seed() -> int:
    """Devuelve la semilla aleatoria configurada."""
    config = load_config()
    return int(config.get("project", {}).get("random_seed", DEFAULT_RANDOM_SEED))


def get_qn_columns() -> list[str]:
    """Lista de columnas QN recodificadas disponibles en el dataset GSHS."""
    return [f"QN{i}" for i in range(6, 59)]


def get_excluded_feature_columns() -> list[str]:
    """Columnas que nunca deben usarse como features."""
    return LEAKAGE_COLUMNS + SURVEY_DESIGN_COLUMNS + [TARGET_IMC, TARGET_MENTAL_HEALTH]


def get_mental_health_target_col(config: dict[str, Any] | None = None) -> str:
    """Devuelve la columna QN activa para el target de salud mental."""
    cfg = config if config is not None else load_config()
    key = str(
        cfg.get("data", {}).get(
            "mental_health_target",
            DEFAULT_MENTAL_HEALTH_TARGET_KEY,
        )
    ).lower()
    return MENTAL_HEALTH_TARGET_MAP.get(key, MENTAL_HEALTH_PRIMARY_COL)
