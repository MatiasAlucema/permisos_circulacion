"""
train.py -- Entrena el modelo de prediccion de permisos y exporta el artefacto.

Integra MLflow para experiment tracking: cada ejecucion registra parametros,
metricas, artefactos y el modelo en el registry.

Uso:
    python src/train.py

Output:
    models/model_pipeline.joblib  -- Pipeline completo (preprocessing + SMOTE + modelo)
    models/metadata.json          -- Metadata del entrenamiento
    mlruns/                       -- Experimentos MLflow (ver con: mlflow ui)
"""

import hashlib
import json
import os
import sys
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Agregar root del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLEAN_FILE, N_SPLITS_CV, RANDOM_SEED, TEST_SIZE

# --- Paths de salida ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model_pipeline.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# --- MLflow config ---
MLFLOW_EXPERIMENT = "permisos-circulacion"
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "mlruns")


def get_data_hash(filepath: str) -> str:
    """Calcula hash MD5 del dataset para trazabilidad."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    """Carga datos limpios y aplica feature engineering."""
    df = pd.read_csv(CLEAN_FILE)
    df["fecha_emision"] = pd.to_datetime(df["fecha_emision"])

    # Feature engineering (misma logica que el notebook 04)
    df["mes"] = df["fecha_emision"].dt.month
    df["dia_semana"] = df["fecha_emision"].dt.dayofweek
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["dias_antiguedad"] = (pd.Timestamp("2025-07-01") - df["fecha_emision"]).dt.days

    df.drop(columns=["id_permiso", "fecha_emision"], inplace=True)

    X = df.drop(columns=["estado"])
    y = (df["estado"] == "Inactivo").astype(int)

    return X, y


def build_pipeline(X: pd.DataFrame) -> ImbPipeline:
    """Construye el pipeline completo: preprocessing + SMOTE + modelo."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
        ]
    )

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_SEED)),
    ])

    return pipeline


def tune_hyperparameters(pipeline: ImbPipeline, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """Optimiza hiperparametros con RandomizedSearchCV."""
    param_dist = {
        "classifier__n_estimators": [50, 100, 200, 300],
        "classifier__max_depth": [5, 10, 15, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=30,
        cv=cv,
        scoring="f1",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"\n[OK] Mejor F1 (CV): {search.best_score_:.4f}")
    print(f"[OK] Mejores parametros: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(model: ImbPipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evalua el modelo en test set y retorna metricas."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    print("\n=== EVALUACION FINAL (Test Set) ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Activo', 'Inactivo'])}")

    return metrics


def save_artifacts(model: ImbPipeline, metrics: dict, best_params: dict, best_cv_f1: float,
                   feature_names: list[str]) -> None:
    """Guarda el modelo y metadata localmente."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Guardar pipeline completo
    joblib.dump(model, MODEL_PATH)
    print(f"\n[OK] Modelo guardado en: {MODEL_PATH}")

    # Guardar metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "pipeline": "StandardScaler + OneHotEncoder + SMOTE + RandomForest",
        "best_cv_f1": round(best_cv_f1, 4),
        "best_params": {k.replace("classifier__", ""): v for k, v in best_params.items()},
        "test_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "features": feature_names,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "cv_folds": N_SPLITS_CV,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] Metadata guardada en: {METADATA_PATH}")


def main():
    print("=" * 60)
    print("  ENTRENAMIENTO - Prediccion de Permisos de Circulacion")
    print("=" * 60)

    # ── Configurar MLflow ──────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"\n[MLflow] Experiment: {MLFLOW_EXPERIMENT}")
    print(f"[MLflow] Tracking URI: {MLFLOW_TRACKING_URI}")

    with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"[MLflow] Run ID: {run.info.run_id}")

        # ── 1. Cargar datos ────────────────────────────────
        print("\n[1/5] Cargando y preparando datos...")
        X, y = load_and_prepare_data()
        print(f"  Dataset: {X.shape[0]} registros, {X.shape[1]} features")
        print(f"  Target: {y.value_counts().to_dict()} (ratio inactivos: {y.mean():.3f})")

        # Log dataset info en MLflow
        data_hash = get_data_hash(CLEAN_FILE)
        mlflow.log_params({
            "dataset_path": os.path.basename(CLEAN_FILE),
            "dataset_hash_md5": data_hash,
            "dataset_rows": X.shape[0],
            "dataset_features": X.shape[1],
            "target_ratio_inactive": round(y.mean(), 4),
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "cv_folds": N_SPLITS_CV,
            "balancing_method": "SMOTE",
        })

        # ── 2. Split ───────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        mlflow.log_params({
            "train_size": X_train.shape[0],
            "test_size_n": X_test.shape[0],
        })

        # ── 3. Construir y optimizar pipeline ──────────────
        print("\n[2/5] Construyendo pipeline y optimizando hiperparametros...")
        pipeline = build_pipeline(X_train)
        best_model, best_params, best_cv_f1 = tune_hyperparameters(pipeline, X_train, y_train)

        # Log hiperparametros del mejor modelo
        clean_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
        mlflow.log_params({f"best_{k}": v for k, v in clean_params.items()})
        mlflow.log_metric("cv_f1_best", best_cv_f1)

        # ── 4. Evaluar ─────────────────────────────────────
        print("\n[3/5] Evaluando modelo en test set...")
        metrics = evaluate_model(best_model, X_test, y_test)

        # Log metricas en MLflow
        for k, v in metrics.items():
            if k != "confusion_matrix":
                mlflow.log_metric(f"test_{k}", v)

        # ── 5. Guardar artefactos locales ──────────────────
        print("\n[4/5] Guardando artefactos locales...")
        feature_names = X.columns.tolist()
        save_artifacts(best_model, metrics, best_params, best_cv_f1, feature_names)

        # ── 6. Registrar modelo en MLflow ──────────────────
        print("\n[5/5] Registrando modelo en MLflow...")

        # Log el artefacto joblib
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(METADATA_PATH, artifact_path="model")

        # Log el modelo sklearn para servir directo desde MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="sklearn_pipeline",
            registered_model_name="permisos-circulacion-model",
            input_example=X_test.iloc[:1],
        )

        # Log features como artefacto de texto
        features_path = os.path.join(MODELS_DIR, "features.txt")
        with open(features_path, "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact(features_path, artifact_path="model")

        print("\n[MLflow] Modelo registrado como: permisos-circulacion-model")
        print(f"[MLflow] Run ID: {run.info.run_id}")
        print(f"[MLflow] Para ver resultados: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
