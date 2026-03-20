# Prediccion de Estado de Permisos de Circulacion

Proyecto end-to-end de **AI & Data Engineering** que aborda el ciclo completo: generacion de datos sinteticos, limpieza, EDA, modelado predictivo, **API REST containerizada**, **experiment tracking**, **explicaciones con IA**, **data drift monitoring**, **SHAP explainability** y **CI/CD**.

## Problema

Predecir si un permiso de circulacion sera **Inactivo** (riesgo de irregularidad/expiracion) o **Activo**, usando variables como tipo de vehiculo, zona, infracciones previas, monto pagado y antiguedad.

## Pipeline

```
data_generation.py -> 02_cleaning -> 03_EDA -> 04_modeling -> train.py -> API REST
     |                    |            |          |           |          |
  CSV bruto          CSV limpio   Insights    Modelo      Artefacto   FastAPI + Docker
  (con errores)      (validado)   y graficos  final       .joblib       |
                                                |           |        /predict/explain (LLM)
                                              SHAP       MLflow     /monitoring/drift (Evidently)
                                                        (tracking)
                                                            |
                                               pytest + ruff + GitHub Actions (CI)
```

## Estructura del repositorio

```
|-- api/
|   |-- main.py                 # API REST -- /predict, /predict/explain, /monitoring/drift
|   |-- schemas.py              # Schemas Pydantic v2 (request/response validation)
|-- src/
|   |-- data_generation.py      # Generador de datos sinteticos con reglas de negocio
|   |-- train.py                # Entrenamiento + MLflow tracking + model registry
|   |-- predict.py              # Modulo de inferencia reutilizable (singleton)
|   |-- evaluate.py             # Evaluacion y comparacion de modelos desde MLflow
|   |-- llm_explainer.py        # Explicaciones con LLM (Gemini / Claude, multi-provider)
|   |-- monitoring.py           # Deteccion de data drift con Evidently AI
|   |-- explainability.py       # SHAP values (explicabilidad local y global)
|-- tests/
|   |-- test_predict.py         # Tests del predictor (11 tests)
|   |-- test_api.py             # Tests de endpoints (10 tests)
|   |-- test_data_quality.py    # Tests de calidad de datos (11 tests)
|-- config/
|   |-- config.py               # Constantes, paths y parametros centralizados
|-- notebooks/
|   |-- 02_data_validation_cleaning.ipynb
|   |-- 03_eda_exploratory_analysis.ipynb
|   |-- 04_feature_engineering_modeling.ipynb
|-- models/                     # Artefactos del modelo (.joblib + metadata.json)
|-- data/
|   |-- raw/                    # Datos brutos (generados)
|   |-- processed/              # Datos limpios
|   |-- reference/              # Datos de referencia para drift detection
|-- .github/workflows/ci.yml   # CI pipeline (lint -> test -> docker build)
|-- Dockerfile                  # Multi-stage build (builder + runtime)
|-- docker-compose.yml          # Orquestacion local
|-- pyproject.toml              # Config de ruff + pytest
|-- .env.example                # Template de variables de entorno
|-- requirements.txt
|-- README.md
```

## Habilidades demostradas

| Area | Tecnicas |
|------|----------|
| **Data Engineering** | Generacion de datos sinteticos, pipeline de limpieza, validacion con assertions |
| **EDA** | Analisis bivariado, heatmaps de interaccion, analisis temporal, correlaciones |
| **ML** | Pipeline con `ColumnTransformer`, StratifiedKFold CV, comparacion de 3 modelos |
| **Desbalance** | SMOTE dentro del pipeline de CV (sin data leakage), class_weight |
| **Tuning** | `RandomizedSearchCV`, analisis de threshold, curvas ROC y Precision-Recall |
| **Model Serving** | FastAPI REST API, Pydantic v2 schemas, prediccion individual y batch |
| **Containerizacion** | Dockerfile multi-stage, docker-compose, health checks |
| **Experiment Tracking** | MLflow tracking (params, metrics, artifacts), Model Registry |
| **LLM Integration** | Gemini / Claude API, prompt engineering, multi-provider architecture |
| **Model Monitoring** | Evidently AI data drift (Wasserstein, Jensen-Shannon), auto-logging |
| **Explainability** | SHAP TreeExplainer, explicabilidad local y global, feature contributions |
| **Testing** | pytest (32+ tests), data quality tests, API tests, predictor tests |
| **CI/CD** | GitHub Actions pipeline: ruff lint -> pytest -> Docker build |
| **Buenas practicas** | Config centralizada, reproducibilidad, .env para secrets, pyproject.toml |

## Decisiones tecnicas clave

- **Datos con senal real**: La variable target depende de las features mediante reglas de negocio, no es aleatoria
- **SMOTE dentro del CV**: Se aplica solo en el fold de entrenamiento usando `imblearn.Pipeline` para evitar data leakage
- **Pipeline serializado completo**: El artefacto `.joblib` incluye preprocessing + SMOTE + modelo
- **Separacion train/predict/serve**: Cada componente es independiente y testeable
- **MLflow experiment tracking**: Cada entrenamiento registra parametros, metricas, artefactos y hash MD5 del dataset
- **LLM multi-provider**: Cambia entre Gemini y Claude con una variable de entorno (`LLM_PROVIDER`)
- **Data drift monitoring**: Cada request se loguea. `/monitoring/drift` compara distribuciones con Evidently AI
- **SHAP explainability**: TreeExplainer descompone predicciones en contribuciones por feature
- **Data contracts**: Tests automatizados validan que los datos cumplen reglas de negocio

## Como ejecutar

### Opcion 1: Local (desarrollo)

```bash
# 1. Clonar e instalar dependencias
git clone https://github.com/MatiasAlucema/data-and-IA-portfolio.git
cd proyecto_datos_circulacion
pip install -r requirements.txt

# 2. Configurar API key (requerida para /predict/explain)
cp .env.example .env
# Editar .env con tu GEMINI_API_KEY o ANTHROPIC_API_KEY

# 3. Generar datos y entrenar
python src/data_generation.py
jupyter notebook  # Ejecutar notebooks 02 -> 03 -> 04
python src/train.py

# 4. Ejecutar tests
python -m pytest tests/ -v

# 5. Lint
ruff check src/ api/ tests/

# 6. Levantar API
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs

# 7. MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
# http://localhost:5000
```

### Opcion 2: Docker

```bash
python src/train.py  # Genera models/
docker compose up --build
# API: http://localhost:8000/docs
```

### Ejemplos de request

```bash
# Prediccion simple
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tipo_vehiculo":"Camion","duracion_dias":20,"zona_circulacion":"Zona A",
       "monto_pagado":85000,"renovacion":false,"infracciones_previas":3,
       "fecha_emision":"2025-03-15"}'

# Prediccion + explicacion IA
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"tipo_vehiculo":"Camion","duracion_dias":20,"zona_circulacion":"Zona A",
       "monto_pagado":85000,"renovacion":false,"infracciones_previas":3,
       "fecha_emision":"2025-03-15"}'

# Verificar drift
curl http://localhost:8000/monitoring/drift
```

## Stack

Python | Pandas | Scikit-learn | FastAPI | Pydantic v2 | Docker | MLflow | Evidently AI | SHAP | Gemini/Claude API | pytest | ruff | GitHub Actions
