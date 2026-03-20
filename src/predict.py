"""
predict.py — Módulo de inferencia reutilizable.

Carga el modelo entrenado y expone funciones de predicción
que son consumidas por la API y otros scripts.
"""

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_pipeline.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "metadata.json")

# Labels
LABELS = {0: "Activo", 1: "Inactivo"}


@dataclass
class PredictionResult:
    """Resultado de una predicción individual."""
    prediction: str
    probability_inactive: float
    probability_active: float
    features_used: dict


class PermitPredictor:
    """Predictor de permisos de circulación.

    Carga el pipeline entrenado (preprocessing + SMOTE + modelo) y expone
    métodos para predecir sobre datos nuevos.
    """

    def __init__(self, model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH):
        self.model = joblib.load(model_path)
        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["features"]

    def predict_single(self, data: dict) -> PredictionResult:
        """Predice el estado de un único permiso.

        Args:
            data: Diccionario con las features del permiso. Debe incluir:
                - tipo_vehiculo (str)
                - duracion_dias (float)
                - zona_circulacion (str)
                - monto_pagado (float)
                - renovacion (bool)
                - infracciones_previas (int)
                - fecha_emision (str, formato YYYY-MM-DD)

        Returns:
            PredictionResult con predicción, probabilidades y features.
        """
        df = self._prepare_input(data)
        proba = self.model.predict_proba(df)[0]
        pred_idx = int(np.argmax(proba))

        return PredictionResult(
            prediction=LABELS[pred_idx],
            probability_inactive=round(float(proba[1]), 4),
            probability_active=round(float(proba[0]), 4),
            features_used=df.iloc[0].to_dict(),
        )

    def predict_batch(self, data_list: list[dict]) -> list[PredictionResult]:
        """Predice el estado de múltiples permisos."""
        return [self.predict_single(d) for d in data_list]

    def _prepare_input(self, data: dict) -> pd.DataFrame:
        """Aplica feature engineering al input (misma lógica que train.py)."""
        df = pd.DataFrame([data])

        # Feature engineering
        if "fecha_emision" in df.columns:
            df["fecha_emision"] = pd.to_datetime(df["fecha_emision"])
            df["mes"] = df["fecha_emision"].dt.month
            df["dia_semana"] = df["fecha_emision"].dt.dayofweek
            df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
            df["dias_antiguedad"] = (pd.Timestamp("2025-07-01") - df["fecha_emision"]).dt.days
            df.drop(columns=["fecha_emision"], inplace=True)

        # Asegurar que solo usamos las features esperadas
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Feature faltante: '{col}'")

        return df[self.feature_names]

    def get_model_info(self) -> dict:
        """Retorna metadata del modelo cargado."""
        return {
            "model_type": self.metadata["model_type"],
            "pipeline": self.metadata["pipeline"],
            "trained_at": self.metadata["trained_at"],
            "test_metrics": self.metadata["test_metrics"],
            "features": self.feature_names,
        }


# Singleton — se carga una vez y se reutiliza
_predictor: PermitPredictor | None = None


def get_predictor() -> PermitPredictor:
    """Obtiene la instancia singleton del predictor."""
    global _predictor
    if _predictor is None:
        _predictor = PermitPredictor()
    return _predictor
