"""Entrenamiento de modelos de machine learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import get_random_seed
from labels import get_label
from models.predict import save_model


@dataclass
class RegressionResults:
    """Resultados de evaluación de regresión."""

    model_name: str
    rmse: float
    r2: float
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_r2_mean: float
    cv_r2_std: float
    residuals: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    best_model: Any = field(repr=False, default=None)


@dataclass
class TunedModelResults:
    """Resultados de evaluación en test de un modelo con hiperparámetros ajustados."""

    task: str
    model_name: str
    best_params: dict[str, Any]
    rmse: float | None = None
    r2: float | None = None
    f1_minority: float | None = None
    auc_roc: float | None = None
    accuracy: float | None = None
    confusion: np.ndarray | None = None
    classification_report: str | None = None
    best_model: Any = field(repr=False, default=None)
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class ClassificationResults:
    """Resultados de evaluación de clasificación."""

    model_name: str
    f1_minority: float
    auc_roc: float
    accuracy: float
    confusion: np.ndarray
    classification_report: str
    cv_f1_mean: float
    cv_f1_std: float
    cv_auc_mean: float
    cv_auc_std: float
    y_test: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    y_proba: np.ndarray | None = field(repr=False, default=None)
    fpr: np.ndarray | None = field(repr=False, default=None)
    tpr: np.ndarray | None = field(repr=False, default=None)
    best_model: Any = field(repr=False, default=None)
    feature_importances: dict[str, float] = field(default_factory=dict)


def build_regression_models(random_state: int) -> dict[str, BaseEstimator]:
    """Construye estimadores de regresión para comparación."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def build_classification_models(random_state: int) -> dict[str, BaseEstimator]:
    """Construye estimadores de clasificación con manejo de desbalance."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _make_pipeline(
    preprocessor: Any,
    estimator: BaseEstimator,
    use_smote: bool = False,
) -> Pipeline | ImbPipeline:
    """Crea pipeline de preprocesamiento + estimador (+ SMOTE opcional)."""
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=get_random_seed())))
    steps.append(("model", estimator))
    if use_smote:
        return ImbPipeline(steps)
    return Pipeline(steps)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    estimator: BaseEstimator,
    output_path: Path | None = None,
    use_smote: bool = False,
) -> Pipeline | ImbPipeline:
    """
    Entrena un modelo con pipeline de preprocesamiento.

    Args:
        X: Features de entrenamiento.
        y: Variable objetivo.
        preprocessor: ColumnTransformer de preprocesamiento.
        estimator: Estimador de scikit-learn / XGBoost.
        output_path: Ruta opcional para guardar el artefacto.
        use_smote: Si True, aplica SMOTE en el pipeline (clasificación).

    Returns:
        Pipeline entrenado.
    """
    pipeline = _make_pipeline(preprocessor, estimator, use_smote=use_smote)
    pipeline.fit(X, y)

    if output_path is not None:
        save_model(pipeline, output_path)

    return pipeline


def tune_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Ajuste básico de hiperparámetros para Random Forest de regresión."""
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(random_state=random_state, n_jobs=-1),
            ),
        ]
    )
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [3, 5],
    }
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_classification_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    random_state: int,
    use_smote: bool = True,
) -> tuple[Pipeline | ImbPipeline, dict[str, Any]]:
    """Ajuste básico de hiperparámetros para clasificación con SMOTE."""
    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = _make_pipeline(preprocessor, estimator, use_smote=use_smote)
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [3, 5],
    }
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_regression(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int | None = None,
) -> list[RegressionResults]:
    """
    Entrena y evalúa modelos de regresión con validación cruzada.

    Métricas: RMSE (minimizar) y R² (maximizar).
    """
    seed = random_state if random_state is not None else get_random_seed()
    mask = y.notna()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=seed
    )

    results: list[RegressionResults] = []
    models = build_regression_models(seed)

    for name, estimator in models.items():
        preproc = clone(preprocessor)
        pipeline = train_model(X_train, y_train, preproc, estimator)
        y_pred = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        residuals = y_test.values - y_pred

        cv_neg_rmse = cross_val_score(
            pipeline, X_clean, y_clean, cv=cv_folds, scoring="neg_root_mean_squared_error"
        )
        cv_r2 = cross_val_score(pipeline, X_clean, y_clean, cv=cv_folds, scoring="r2")

        results.append(
            RegressionResults(
                model_name=name,
                rmse=rmse,
                r2=r2,
                cv_rmse_mean=float(-cv_neg_rmse.mean()),
                cv_rmse_std=float(cv_neg_rmse.std()),
                cv_r2_mean=float(cv_r2.mean()),
                cv_r2_std=float(cv_r2.std()),
                residuals=residuals,
                y_test=y_test.values,
                y_pred=y_pred,
                best_model=pipeline,
            )
        )

    return results


def evaluate_classification(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int | None = None,
    use_smote: bool = True,
) -> list[ClassificationResults]:
    """
    Entrena y evalúa modelos de clasificación con manejo de desbalance.

    Métricas principales: F1 de la clase minoritaria y AUC-ROC.
    """
    seed = random_state if random_state is not None else get_random_seed()
    mask = y.notna()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        test_size=test_size,
        random_state=seed,
        stratify=y_clean,
    )

    results: list[ClassificationResults] = []
    models = build_classification_models(seed)

    for name, estimator in models.items():
        apply_smote = use_smote and name != "logistic_regression"
        preproc = clone(preprocessor)
        pipeline = train_model(
            X_train, y_train, preproc, estimator, use_smote=apply_smote
        )
        y_pred = pipeline.predict(X_test)

        y_proba = None
        fpr, tpr = None, None
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)

        minority_label = int(y_clean.value_counts().idxmin())
        f1_min = float(f1_score(y_test, y_pred, pos_label=minority_label))
        auc_val = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else 0.0
        accuracy = float((y_pred == y_test).mean())
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)

        cv_f1 = cross_val_score(pipeline, X_clean, y_clean, cv=cv_folds, scoring="f1")
        cv_auc = cross_val_score(
            pipeline, X_clean, y_clean, cv=cv_folds, scoring="roc_auc"
        )

        importances = extract_feature_importances(pipeline)

        results.append(
            ClassificationResults(
                model_name=name,
                f1_minority=f1_min,
                auc_roc=auc_val,
                accuracy=accuracy,
                confusion=cm,
                classification_report=report,
                cv_f1_mean=float(cv_f1.mean()),
                cv_f1_std=float(cv_f1.std()),
                cv_auc_mean=float(cv_auc.mean()),
                cv_auc_std=float(cv_auc.std()),
                y_test=y_test.values,
                y_pred=y_pred,
                y_proba=y_proba,
                fpr=fpr,
                tpr=tpr,
                best_model=pipeline,
                feature_importances=importances,
            )
        )

    return results


def _get_transformed_feature_names(pipeline: Pipeline | ImbPipeline) -> list[str]:
    """Obtiene nombres de features tras preprocesamiento (incluye One-Hot)."""
    preprocessor = pipeline.named_steps["preprocessor"]
    if hasattr(preprocessor, "get_feature_names_out"):
        return list(preprocessor.get_feature_names_out())
    return []


def extract_feature_importances(
    pipeline: Pipeline | ImbPipeline,
) -> dict[str, float]:
    """Extrae importancia de características del modelo entrenado."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        return {}

    feature_names = _get_transformed_feature_names(pipeline)
    if not feature_names or len(importances) != len(feature_names):
        return {}

    pairs = sorted(
        zip(feature_names, importances, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return {name: float(imp) for name, imp in pairs[:15]}


def evaluate_tuned_regression(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    random_state: int,
    test_size: float = 0.2,
) -> TunedModelResults:
    """Ajusta hiperparámetros y evalúa Random Forest en conjunto de test."""
    mask = y.notna()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state
    )
    tuned_model, best_params = tune_regression_model(
        X_train, y_train, clone(preprocessor), random_state
    )
    y_pred = tuned_model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return TunedModelResults(
        task="regression",
        model_name="random_forest_tuned",
        best_params=best_params,
        rmse=rmse,
        r2=r2,
        best_model=tuned_model,
        feature_importances=extract_feature_importances(tuned_model),
    )


def evaluate_tuned_classification(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: Any,
    random_state: int,
    test_size: float = 0.2,
    use_smote: bool = True,
) -> TunedModelResults:
    """Ajusta hiperparámetros y evalúa Random Forest+SMOTE en conjunto de test."""
    mask = y.notna()
    X_clean = X.loc[mask]
    y_clean = y.loc[mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clean,
    )
    tuned_model, best_params = tune_classification_model(
        X_train, y_train, clone(preprocessor), random_state, use_smote=use_smote
    )
    y_pred = tuned_model.predict(X_test)
    y_proba = tuned_model.predict_proba(X_test)[:, 1]
    minority_label = int(y_clean.value_counts().idxmin())

    return TunedModelResults(
        task="classification",
        model_name="random_forest_smote_tuned",
        best_params=best_params,
        f1_minority=float(f1_score(y_test, y_pred, pos_label=minority_label)),
        auc_roc=float(roc_auc_score(y_test, y_proba)),
        accuracy=float((y_pred == y_test).mean()),
        confusion=confusion_matrix(y_test, y_pred),
        classification_report=classification_report(y_test, y_pred, digits=3),
        best_model=tuned_model,
        feature_importances=extract_feature_importances(tuned_model),
    )


def format_feature_importances_for_report(
    importances: dict[str, float],
) -> dict[str, float]:
    """Traduce nombres técnicos de features a etiquetas legibles para el informe."""
    translated: dict[str, float] = {}
    for name, value in importances.items():
        # One-Hot: num__QN6 o cat__Q1_3
        base = name.split("__", 1)[-1] if "__" in name else name
        base_var = base.split("_")[0] if base.startswith("Q") else base
        label = get_label(base_var)
        if base != base_var:
            label = f"{label} ({base})"
        translated[label] = value
    return translated


def select_best_regression(results: list[RegressionResults]) -> RegressionResults:
    """Selecciona el mejor modelo de regresión por RMSE en test."""
    return min(results, key=lambda r: r.rmse)


def select_best_classification(
    results: list[ClassificationResults],
) -> ClassificationResults:
    """Selecciona el mejor modelo de clasificación por F1 minoritaria."""
    return max(results, key=lambda r: r.f1_minority)
