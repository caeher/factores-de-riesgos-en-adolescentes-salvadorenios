# Sistema de Predicción de Factores de Riesgo en Adolescentes Salvadoreños (GSHS 2013)

Pipeline de Machine Learning para el desafío UES/MINSAL sobre la Encuesta Global de Salud Escolar (GSHS) 2013 de El Salvador.

## Requisitos

- Python 3.13+
- PowerShell (Windows)

## Setup rápido

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD\src"
pip install -r requirements.txt
```

Coloca el dataset en `data/raw/SLV2013_Public_Use.csv`.

## Ejecución del pipeline

```powershell
# Pipeline completo (EDA + entrenamiento + figuras + métricas)
python scripts/train.py

# Sin ajuste de hiperparámetros (más rápido)
python scripts/train.py --skip-tuning
```

**Salidas generadas:**
- `data/processed/gshs_processed.csv` — datos limpios con IMC y Riesgo_Salud_Mental
- `models/regression_imc.joblib` — mejor modelo de regresión
- `models/classification_mental_health.joblib` — mejor modelo de clasificación
- `reports/figures/` — visualizaciones EDA y de evaluación
- `reports/metrics.json` — métricas consolidadas

## Notebook de exploración

```powershell
jupyter lab notebooks/01_eda_modelado.ipynb
```

Kernel: **Python (fras)**

## Estructura del proyecto

```
fras/
├── configs/              # Configuración YAML
├── data/raw/             # SLV2013_Public_Use.csv
├── data/processed/       # Datos procesados
├── notebooks/            # 01_eda_modelado.ipynb
├── src/
│   ├── config.py         # Constantes (centinela, targets, leakage)
│   ├── data/             # Carga y preprocesamiento
│   ├── features/         # Feature engineering (QN, sin leakage)
│   ├── models/           # Entrenamiento, evaluación, inferencia
│   └── visualization/    # Gráficos EDA y evaluación
├── models/               # Artefactos .joblib
├── reports/
│   ├── figures/          # Figuras PNG
│   ├── metrics.json
│   └── informe_ieee/     # Informe técnico IEEE (LaTeX)
├── scripts/train.py      # CLI del pipeline
└── tests/                # Tests unitarios
```

## Tareas del desafío

| Tarea | Target | Modelos | Métricas |
|-------|--------|---------|----------|
| A — Regresión IMC | `IMC = Q5 / Q4²` | Linear Regression, Random Forest | RMSE, R², residuos |
| B — Clasificación | `Riesgo_Salud_Mental` (QN24 principal, QN22 alternativo) | Logistic Regression, RF+SMOTE, XGBoost | F1 minoritaria, AUC-ROC |

## Informe IEEE

El informe técnico para el Ministro de Salud está en `reports/informe_ieee/informe.tex`.

```powershell
cd reports/informe_ieee
pdflatex informe.tex
pdflatex informe.tex
```

Requiere una distribución LaTeX con `IEEEtran` y soporte UTF-8.

## Comandos útiles

```powershell
pytest
ruff check src tests
```

## Decisiones técnicas clave

1. **Centinela SPSS** (`1.79e+308`) → `np.nan` al cargar
2. **Sin data leakage**: Q4, Q5 excluidos de features de regresión IMC
3. **QN vs Q**: se usan recodificaciones QN para evitar colinealidad
4. **Target salud mental**: QN24 (ideación suicida) principal; QN22 (soledad) como escenario alternativo
5. **Desbalance**: `class_weight='balanced'` + SMOTE; F1 minoritaria como métrica principal

## Licencia

MIT — ver [LICENSE](LICENSE).
