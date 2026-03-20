"""
monitoring.py -- Deteccion de data drift usando Evidently AI.

Compara los datos de produccion (requests a la API) contra los datos de
entrenamiento (referencia) para detectar cambios en la distribucion.

Conceptos:
- Data drift: las distribuciones de features cambian respecto al entrenamiento
- Si hay drift, el modelo puede perder performance sin que lo sepamos
- Evidently calcula tests estadisticos (K-S, chi-square) por feature

Uso:
    from src.monitoring import DriftMonitor
    monitor = DriftMonitor()
    monitor.log_prediction(features_dict)       # Loguear cada request
    report = monitor.check_drift()              # Generar reporte de drift
"""

import json
import os
import sys
from datetime import datetime
from threading import Lock

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CLEAN_FILE

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_PATH = os.path.join(PROJECT_ROOT, "data", "reference", "reference_data.csv")

# Features a monitorear (numericas y categoricas del modelo)
NUMERIC_FEATURES = ["duracion_dias", "monto_pagado", "infracciones_previas"]
CATEGORICAL_FEATURES = ["tipo_vehiculo", "zona_circulacion"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_reference_data() -> pd.DataFrame:
    """Crea el dataset de referencia a partir de los datos de entrenamiento.

    Este dataset se usa como baseline para comparar contra datos de produccion.
    Solo guarda las features relevantes (no el target ni IDs).
    """
    df = pd.read_csv(CLEAN_FILE)
    ref = df[ALL_FEATURES].copy()

    # Guardar
    os.makedirs(os.path.dirname(REFERENCE_PATH), exist_ok=True)
    ref.to_csv(REFERENCE_PATH, index=False)
    print(f"[OK] Datos de referencia guardados: {REFERENCE_PATH} ({len(ref)} registros)")
    return ref


class DriftMonitor:
    """Monitor de data drift para la API de predicciones.

    Acumula datos de produccion (requests) y los compara contra
    el dataset de referencia para detectar drift.
    """

    def __init__(self, min_samples: int = 30):
        """
        Args:
            min_samples: Minimo de muestras antes de calcular drift.
        """
        self.min_samples = min_samples
        self._production_data: list[dict] = []
        self._lock = Lock()

        # Cargar datos de referencia
        if os.path.exists(REFERENCE_PATH):
            self.reference_data = pd.read_csv(REFERENCE_PATH)
        else:
            print("[WARN] Datos de referencia no encontrados. Ejecute create_reference_data().")
            self.reference_data = None

    def log_prediction(self, features: dict) -> None:
        """Registra las features de un request de produccion.

        Se llama desde cada endpoint /predict o /predict/explain.
        """
        # Extraer solo las features que monitoreamos
        record = {f: features.get(f) for f in ALL_FEATURES}
        with self._lock:
            self._production_data.append(record)

    def get_production_count(self) -> int:
        """Retorna cuantos requests se han acumulado."""
        return len(self._production_data)

    def check_drift(self) -> dict:
        """Ejecuta la deteccion de drift comparando produccion vs referencia.

        Returns:
            Dict con:
            - drift_detected (bool): si hay drift significativo
            - total_features: total de features analizadas
            - drifted_features: lista de features con drift
            - drift_share: proporcion de features con drift
            - details: metricas por feature
            - production_samples: cuantas muestras de produccion se usaron
            - reference_samples: cuantas muestras de referencia
            - checked_at: timestamp
        """
        if self.reference_data is None:
            return {"error": "Datos de referencia no disponibles. Ejecute create_reference_data()."}

        with self._lock:
            if len(self._production_data) < self.min_samples:
                return {
                    "error": f"Insuficientes datos de produccion: {len(self._production_data)}/{self.min_samples} minimo.",
                    "production_samples": len(self._production_data),
                }
            prod_df = pd.DataFrame(self._production_data)

        # Ejecutar reporte de drift con Evidently
        report = Report([DataDriftPreset()])
        snapshot = report.run(
            reference_data=self.reference_data[ALL_FEATURES],
            current_data=prod_df[ALL_FEATURES],
        )

        # Parsear resultado
        raw = json.loads(snapshot.json())
        metrics = raw.get("metrics", [])

        # Primer metrica: DriftedColumnsCount (resumen global)
        drift_share = 0.0
        details = []

        for m in metrics:
            metric_name = m.get("metric_name", "")
            if "DriftedColumnsCount" in metric_name:
                drift_share = float(m["value"].get("share", 0))
            elif "ValueDrift" in metric_name:
                # Per-column drift metric
                # Evidently 0.7 usa distancias (Wasserstein, Jensen-Shannon)
                # Drift = value > threshold (no p-value)
                config = m.get("config", {})
                column = config.get("column", "unknown")
                method = config.get("method", "unknown")
                threshold = config.get("threshold", 0.1)
                distance = float(m.get("value", 0.0))
                details.append(
                    {
                        "feature": column,
                        "method": method,
                        "distance": round(distance, 6),
                        "threshold": threshold,
                        "drift_detected": distance > threshold,
                    }
                )

        drifted_features = [d["feature"] for d in details if d["drift_detected"]]

        return {
            "drift_detected": len(drifted_features) > 0,
            "total_features": len(ALL_FEATURES),
            "drifted_features": drifted_features,
            "drifted_count": len(drifted_features),
            "drift_share": round(drift_share, 4),
            "details": details,
            "production_samples": len(prod_df),
            "reference_samples": len(self.reference_data),
            "checked_at": datetime.now().isoformat(),
        }

    def clear_production_data(self) -> int:
        """Limpia los datos de produccion acumulados. Retorna cuantos se borraron."""
        with self._lock:
            count = len(self._production_data)
            self._production_data.clear()
        return count


# Singleton
_monitor: DriftMonitor | None = None


def get_monitor() -> DriftMonitor:
    """Obtiene la instancia singleton del monitor."""
    global _monitor
    if _monitor is None:
        _monitor = DriftMonitor()
    return _monitor


if __name__ == "__main__":
    # Crear datos de referencia
    create_reference_data()
