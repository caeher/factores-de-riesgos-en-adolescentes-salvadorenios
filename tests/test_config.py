"""Tests de humo para configuración y rutas del proyecto."""

from __future__ import annotations

from config import __version__, get_config_path, get_project_paths, load_config


def test_import_config_module() -> None:
    """El módulo config debe ser importable."""
    assert __version__ == "0.1.0"


def test_project_paths_exist() -> None:
    """Las rutas principales del proyecto deben existir."""
    paths = get_project_paths()

    assert paths.root.is_dir()
    assert paths.data_raw.is_dir()
    assert paths.data_interim.is_dir()
    assert paths.data_processed.is_dir()
    assert paths.data_external.is_dir()
    assert paths.models.is_dir()
    assert paths.reports_figures.is_dir()
    assert paths.configs.is_dir()
    assert paths.notebooks.is_dir()


def test_config_path_exists() -> None:
    """Debe existir al menos el archivo de configuración activo o de ejemplo."""
    config_path = get_config_path()
    assert config_path.exists()
    assert config_path.suffix == ".yaml"


def test_load_config_returns_dict() -> None:
    """load_config debe devolver un diccionario con claves esperadas."""
    config = load_config()

    assert isinstance(config, dict)
    assert "project" in config
    assert "paths" in config
    assert config["project"]["random_seed"] == 42
