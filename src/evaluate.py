"""
evaluate.py -- Evalua un modelo registrado en MLflow contra el test set.

Permite comparar diferentes versiones del modelo cargandolas desde el
MLflow Model Registry, sin depender del artefacto joblib local.

Uso:
    python src/evaluate.py                          # Evalua el ultimo modelo registrado
    python src/evaluate.py --run-id <RUN_ID>        # Evalua un run especifico
    python src/evaluate.py --compare                # Compara todos los runs del experimento
"""

import argparse
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Agregar root del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RANDOM_SEED, TEST_SIZE
from src.train import load_and_prepare_data

# --- MLflow config ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_EXPERIMENT = "permisos-circulacion"


def evaluate_run(run_id: str, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evalua un modelo de un run especifico de MLflow."""
    print(f"\n--- Evaluando run: {run_id[:8]}... ---")

    # Cargar modelo desde MLflow
    model_uri = f"runs:/{run_id}/sklearn_pipeline"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "run_id": run_id[:8],
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print(classification_report(y_test, y_pred, target_names=["Activo", "Inactivo"]))
    return metrics


def compare_runs(X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Compara todos los runs del experimento en una tabla."""
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        print("[ERROR] Experimento no encontrado. Ejecute train.py primero.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if runs.empty:
        print("[ERROR] No hay runs registrados.")
        return

    print(f"\nExperimento: {MLFLOW_EXPERIMENT}")
    print(f"Total runs: {len(runs)}")

    # Evaluar cada run
    results = []
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        try:
            metrics = evaluate_run(run_id, X_test, y_test)
            # Agregar parametros del run
            metrics["started"] = str(row.get("start_time", "N/A"))[:19]
            n_est = row.get("params.best_n_estimators", "?")
            depth = row.get("params.best_max_depth", "?")
            metrics["params"] = f"n_est={n_est}, depth={depth}"
            results.append(metrics)
        except Exception as e:
            print(f"  [SKIP] Run {run_id[:8]}: {e}")

    if not results:
        print("[ERROR] No se pudo evaluar ningun run.")
        return

    # Tabla comparativa
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("  COMPARACION DE MODELOS")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # Mejor modelo
    best = df_results.loc[df_results["f1_score"].idxmax()]
    print(f"\n[BEST] Mejor F1: {best['f1_score']} (run: {best['run_id']})")


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelos desde MLflow")
    parser.add_argument("--run-id", type=str, help="Run ID especifico a evaluar")
    parser.add_argument("--compare", action="store_true", help="Comparar todos los runs")
    args = parser.parse_args()

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Preparar datos de test (misma logica que train.py)
    print("[1/2] Cargando datos de test...")
    from sklearn.model_selection import train_test_split
    X, y = load_and_prepare_data()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Test set: {X_test.shape[0]} registros")

    if args.compare:
        print("\n[2/2] Comparando todos los runs...")
        compare_runs(X_test, y_test)
    elif args.run_id:
        print(f"\n[2/2] Evaluando run: {args.run_id}...")
        metrics = evaluate_run(args.run_id, X_test, y_test)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        # Evaluar el ultimo run
        print("\n[2/2] Evaluando ultimo run...")
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if experiment is None:
            print("[ERROR] Experimento no encontrado. Ejecute train.py primero.")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            print("[ERROR] No hay runs registrados.")
            return

        run_id = runs.iloc[0]["run_id"]
        metrics = evaluate_run(run_id, X_test, y_test)
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
