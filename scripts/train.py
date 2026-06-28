#!/usr/bin/env python
"""CLI de entrada para el pipeline de entrenamiento GSHS 2013."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (
    MENTAL_HEALTH_ALTERNATIVE_COL,
    MENTAL_HEALTH_PRIMARY_COL,
    TARGET_IMC,
    TARGET_MENTAL_HEALTH,
    get_mental_health_target_col,
    get_project_paths,
    load_config,
)
from data.load import load_raw_data
from data.preprocess import build_mental_health_target, preprocess_data
from features.build import build_features, get_target
from models.predict import save_model
from models.train import (
    evaluate_classification,
    evaluate_regression,
    evaluate_tuned_classification,
    evaluate_tuned_regression,
    format_feature_importances_for_report,
    select_best_classification,
    select_best_regression,
)
from sklearn.model_selection import train_test_split
from visualization.plots import (
    plot_age_distribution,
    plot_bmi_by_sex,
    plot_bmi_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_mental_health_by_sex,
    plot_mental_health_prevalence,
    plot_residuals,
    plot_risk_factor_heatmap,
    plot_roc_curve,
    set_plot_style,
)


def parse_args() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline GSHS 2013: regresión IMC + clasificación riesgo salud mental."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Ruta opcional al archivo de configuración YAML.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Omitir ajuste de hiperparámetros (más rápido).",
    )
    return parser.parse_args()


def print_regression_summary(results: list) -> None:
    print("\n=== REGRESIÓN IMC ===")
    for r in results:
        print(
            f"  {r.model_name}: RMSE={r.rmse:.3f} | R²={r.r2:.3f} | "
            f"CV RMSE={r.cv_rmse_mean:.3f}±{r.cv_rmse_std:.3f} | "
            f"CV R²={r.cv_r2_mean:.3f}±{r.cv_r2_std:.3f}"
        )


def print_classification_summary(results: list) -> None:
    print("\n=== CLASIFICACIÓN RIESGO SALUD MENTAL ===")
    for r in results:
        print(
            f"  {r.model_name}: F1(minoritaria)={r.f1_minority:.3f} | "
            f"AUC-ROC={r.auc_roc:.3f} | Accuracy={r.accuracy:.3f} | "
            f"CV F1={r.cv_f1_mean:.3f}±{r.cv_f1_std:.3f}"
        )


def run_classification_scenario(
    df,
    target_col: str,
    scenario_name: str,
    seed: int,
    test_size: float,
    cv_folds: int,
    skip_tuning: bool,
    models_dir: Path,
    figures_dir: Path,
) -> dict:
    """Ejecuta clasificación para un escenario de target (QN24 o QN22)."""
    df_scenario = df.copy()
    df_scenario[TARGET_MENTAL_HEALTH] = build_mental_health_target(
        df_scenario, primary_col=target_col
    )

    X_clf, clf_cols, clf_preprocessor = build_features(df_scenario, "classification")
    y_clf = get_target(df_scenario, "classification")
    clf_results = evaluate_classification(
        X_clf,
        y_clf,
        clf_preprocessor,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=seed,
    )
    print(f"\n--- Escenario: {scenario_name} ({target_col}) ---")
    print_classification_summary(clf_results)
    best_clf = select_best_classification(clf_results)
    print(f"  Mejor modelo clasificación: {best_clf.model_name}")

    suffix = target_col.lower()
    save_model(
        best_clf.best_model,
        models_dir / f"classification_mental_health_{suffix}.joblib",
    )
    plot_confusion_matrix(
        best_clf.confusion,
        figures_dir / f"classification_confusion_matrix_{suffix}.png",
        title=f"Matriz de confusión — {scenario_name}",
    )
    if best_clf.fpr is not None and best_clf.tpr is not None:
        plot_roc_curve(
            best_clf.fpr,
            best_clf.tpr,
            best_clf.auc_roc,
            figures_dir / f"classification_roc_curve_{suffix}.png",
        )
    if best_clf.feature_importances:
        translated = format_feature_importances_for_report(best_clf.feature_importances)
        plot_feature_importance(
            translated,
            figures_dir / f"classification_feature_importance_{suffix}.png",
            title=f"Predictores de riesgo — {scenario_name}",
            use_spanish_labels=False,
        )

    tuned_metrics = None
    if not skip_tuning:
        tuned = evaluate_tuned_classification(
            X_clf, y_clf, clf_preprocessor, seed, test_size=test_size
        )
        save_model(
            tuned.best_model,
            models_dir / f"classification_mental_health_{suffix}_tuned.joblib",
        )
        print(f"  Tuned RF+SMOTE: F1={tuned.f1_minority:.3f} | AUC={tuned.auc_roc:.3f}")
        print(f"  Hiperparámetros: {tuned.best_params}")
        tuned_metrics = {
            "model_name": tuned.model_name,
            "f1_minority": tuned.f1_minority,
            "auc_roc": tuned.auc_roc,
            "accuracy": tuned.accuracy,
            "best_params": tuned.best_params,
            "top_features": format_feature_importances_for_report(
                tuned.feature_importances
            ),
        }

    return {
        "target_column": target_col,
        "scenario_name": scenario_name,
        "best_model": best_clf.model_name,
        "f1_minority": best_clf.f1_minority,
        "auc_roc": best_clf.auc_roc,
        "accuracy": best_clf.accuracy,
        "cv_f1_mean": best_clf.cv_f1_mean,
        "cv_auc_mean": best_clf.cv_auc_mean,
        "feature_columns": clf_cols,
        "top_features": format_feature_importances_for_report(
            best_clf.feature_importances
        ),
        "tuned": tuned_metrics,
        "prevalence": float(df_scenario[TARGET_MENTAL_HEALTH].mean()),
    }


def main() -> int:
    """Ejecuta el pipeline completo de entrenamiento y evaluación."""
    args = parse_args()
    paths = get_project_paths()
    config = load_config()
    seed = config.get("project", {}).get("random_seed", 42)
    test_size = config.get("training", {}).get("test_size", 0.2)
    cv_folds = config.get("training", {}).get("cross_validation_folds", 5)
    active_target = get_mental_health_target_col(config)

    set_plot_style()
    figures_dir = paths.reports_figures
    models_dir = paths.models
    metrics_path = paths.reports / "metrics.json"

    print(f"Proyecto: {paths.root}")
    print("Cargando datos GSHS 2013...")
    df_raw = load_raw_data()
    print(f"  Registros cargados: {len(df_raw):,}")

    print("Preprocesando...")
    df = preprocess_data(df_raw)
    processed_path = paths.data_processed / "gshs_processed.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"  Datos procesados guardados en: {processed_path}")

    # EDA visualizaciones
    print("Generando visualizaciones EDA...")
    plot_age_distribution(df, figures_dir / "eda_age_distribution.png")
    plot_bmi_distribution(df, figures_dir / "eda_bmi_distribution.png")
    plot_bmi_by_sex(df, figures_dir / "eda_bmi_by_sex.png")
    plot_mental_health_prevalence(df, figures_dir / "eda_mental_health_prevalence.png")
    plot_mental_health_by_sex(df, figures_dir / "eda_mental_health_by_sex.png")

    risk_cols = [
        "QN20",
        "QN35",
        "QN38",
        "QN53",
        "QN55",
        "QN56",
        "QN57",
        "QN52",
    ]
    plot_risk_factor_heatmap(
        df, risk_cols, figures_dir / "eda_risk_factor_heatmap.png"
    )

    valid_imc = df[TARGET_IMC].notna().sum()
    valid_risk = df[TARGET_MENTAL_HEALTH].notna().sum()
    risk_rate = df[TARGET_MENTAL_HEALTH].mean()
    print(f"  IMC válidos: {valid_imc:,} | Riesgo mental válidos: {valid_risk:,}")
    print(f"  Prevalencia riesgo mental ({active_target}): {risk_rate:.1%}")

    # --- Regresión IMC ---
    print("\nEntrenando modelos de regresión (IMC)...")
    X_reg, reg_cols, reg_preprocessor = build_features(df, "regression")
    y_reg = get_target(df, "regression")
    reg_results = evaluate_regression(
        X_reg,
        y_reg,
        reg_preprocessor,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=seed,
    )
    print_regression_summary(reg_results)
    best_reg = select_best_regression(reg_results)
    print(f"  Mejor modelo regresión: {best_reg.model_name}")

    save_model(best_reg.best_model, models_dir / "regression_imc.joblib")
    plot_residuals(
        best_reg.y_test,
        best_reg.y_pred,
        figures_dir / "regression_residuals.png",
    )

    tuned_reg_metrics = None
    if not args.skip_tuning:
        tuned_reg = evaluate_tuned_regression(
            X_reg, y_reg, reg_preprocessor, seed, test_size=test_size
        )
        save_model(tuned_reg.best_model, models_dir / "regression_imc_tuned.joblib")
        print(
            f"  Tuned RF: RMSE={tuned_reg.rmse:.3f} | R²={tuned_reg.r2:.3f} | "
            f"params={tuned_reg.best_params}"
        )
        tuned_reg_metrics = {
            "model_name": tuned_reg.model_name,
            "rmse": tuned_reg.rmse,
            "r2": tuned_reg.r2,
            "best_params": tuned_reg.best_params,
        }

    # --- Clasificación: escenario principal (QN24) y alternativo (QN22) ---
    clf_primary = run_classification_scenario(
        df,
        MENTAL_HEALTH_PRIMARY_COL,
        "Ideación suicida (QN24)",
        seed,
        test_size,
        cv_folds,
        args.skip_tuning,
        models_dir,
        figures_dir,
    )
    clf_alternative = run_classification_scenario(
        df,
        MENTAL_HEALTH_ALTERNATIVE_COL,
        "Soledad (QN22)",
        seed,
        test_size,
        cv_folds,
        args.skip_tuning,
        models_dir,
        figures_dir,
    )

    # Compatibilidad: copiar modelo principal al nombre legacy
    primary_model = models_dir / "classification_mental_health_qn24.joblib"
    legacy_model = models_dir / "classification_mental_health.joblib"
    if primary_model.exists():
        legacy_model.write_bytes(primary_model.read_bytes())

    metrics = {
        "regression": {
            "best_model": best_reg.model_name,
            "rmse": best_reg.rmse,
            "r2": best_reg.r2,
            "cv_rmse_mean": best_reg.cv_rmse_mean,
            "cv_r2_mean": best_reg.cv_r2_mean,
            "feature_columns": reg_cols,
            "tuned": tuned_reg_metrics,
            "interpretation": (
                "R² bajo es esperable: el IMC depende principalmente de peso/altura, "
                "excluidos del modelo para evitar data leakage. Las variables de "
                "comportamiento aportan señal limitada pero interpretable."
            ),
        },
        "classification": {
            "active_target": active_target,
            "primary_scenario": clf_primary,
            "alternative_scenario": clf_alternative,
        },
        "eda": {
            "n_records": len(df),
            "valid_imc": int(valid_imc),
            "valid_mental_health": int(valid_risk),
            "mental_health_prevalence": float(risk_rate)
            if not np.isnan(risk_rate)
            else None,
            "figures": [
                "eda_age_distribution.png",
                "eda_bmi_distribution.png",
                "eda_bmi_by_sex.png",
                "eda_mental_health_prevalence.png",
                "eda_mental_health_by_sex.png",
                "eda_risk_factor_heatmap.png",
                "regression_residuals.png",
                "classification_confusion_matrix_qn24.png",
                "classification_roc_curve_qn24.png",
                "classification_feature_importance_qn24.png",
                "classification_confusion_matrix_qn22.png",
                "classification_roc_curve_qn22.png",
                "classification_feature_importance_qn22.png",
            ],
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nMétricas guardadas en: {metrics_path}")
    print(f"Modelos en: {models_dir}")
    print(f"Figuras en: {figures_dir}")
    print("\nPipeline completado exitosamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
