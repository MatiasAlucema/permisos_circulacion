"""
explainability.py -- Explicabilidad del modelo con SHAP values.

SHAP (SHapley Additive exPlanations) descompone cada prediccion en la
contribucion de cada feature, permitiendo entender POR QUE el modelo
predice lo que predice.

Ofrece:
- Explicabilidad global: que features importan mas en general
- Explicabilidad local: por que se predijo X para un caso especifico

Uso:
    from src.explainability import SHAPExplainer
    explainer = SHAPExplainer()
    local = explainer.explain_single(features_dict)
    global_imp = explainer.global_importance()
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RANDOM_SEED, TEST_SIZE

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_pipeline.joblib")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "metadata.json")


class SHAPExplainer:
    """Genera explicaciones SHAP para el modelo de permisos.

    Usa TreeExplainer (optimizado para RandomForest) para calcular
    SHAP values de forma eficiente.
    """

    def __init__(self):
        # Cargar pipeline completo
        self.pipeline = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata["features"]

        # Extraer el clasificador y preprocessor del pipeline
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        # Obtener nombres de features transformadas
        self.transformed_feature_names = [
            name.replace("num__", "").replace("cat__", "")
            for name in self.preprocessor.get_feature_names_out()
        ]

        # Crear explainer de SHAP (TreeExplainer para RF)
        self.explainer = shap.TreeExplainer(self.classifier)

        # Preparar background data para referencia
        self._prepare_background()

    def _prepare_background(self) -> None:
        """Prepara un subset de datos de entrenamiento como background."""
        from sklearn.model_selection import train_test_split

        from src.train import load_and_prepare_data

        X, y = load_and_prepare_data()
        X_train, _, _, _ = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )
        # Transformar con el preprocessor
        self.X_train_transformed = self.preprocessor.transform(X_train)

    def _prepare_input(self, data: dict) -> np.ndarray:
        """Prepara un input individual para SHAP."""
        df = pd.DataFrame([data])

        # Feature engineering (igual que train.py)
        if "fecha_emision" in df.columns:
            df["fecha_emision"] = pd.to_datetime(df["fecha_emision"])
            df["mes"] = df["fecha_emision"].dt.month
            df["dia_semana"] = df["fecha_emision"].dt.dayofweek
            df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
            df["dias_antiguedad"] = (pd.Timestamp("2025-07-01") - df["fecha_emision"]).dt.days
            df.drop(columns=["fecha_emision"], inplace=True)

        df = df[self.feature_names]
        return self.preprocessor.transform(df)

    def explain_single(self, data: dict) -> dict:
        """Genera SHAP explanation para una prediccion individual.

        Args:
            data: Dict con features del permiso (mismo formato que /predict)

        Returns:
            Dict con:
            - shap_values: contribucion de cada feature a la prediccion
            - base_value: valor base (prediccion promedio)
            - prediction_value: prediccion para esta instancia
            - top_positive: features que mas empujan hacia Inactivo
            - top_negative: features que mas empujan hacia Activo
        """
        X_transformed = self._prepare_input(data)
        shap_values = self.explainer.shap_values(X_transformed)

        # Para clasificacion binaria, usar clase 1 (Inactivo)
        # SHAP puede devolver: list[ndarray], ndarray 3D (samples, features, classes)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base = self.explainer.expected_value[1]
        elif shap_values.ndim == 3:
            # Shape: (n_samples, n_features, n_classes) -> clase 1, muestra 0
            sv = shap_values[0, :, 1]
            base = self.explainer.expected_value[1]
        else:
            sv = shap_values[0]
            base = self.explainer.expected_value

        # Crear dict feature -> shap value
        feature_contributions = {}
        for name, value in zip(self.transformed_feature_names, sv):
            feature_contributions[name] = round(float(value), 6)

        # Ordenar por impacto absoluto
        sorted_features = sorted(
            feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Top features que empujan hacia Inactivo (positivas) y Activo (negativas)
        top_positive = [
            {"feature": f, "contribution": v}
            for f, v in sorted_features if v > 0
        ][:5]

        top_negative = [
            {"feature": f, "contribution": v}
            for f, v in sorted_features if v < 0
        ][:5]

        return {
            "shap_values": feature_contributions,
            "base_value": round(float(base), 6),
            "prediction_value": round(float(base + sum(sv)), 6),
            "top_pushing_inactive": top_positive,
            "top_pushing_active": top_negative,
        }

    def global_importance(self) -> dict:
        """Calcula la importancia global de features usando SHAP.

        Usa un sample del training set para calcular SHAP values promedio.

        Returns:
            Dict con feature_importance ordenado por impacto.
        """
        # Usar un sample para eficiencia
        sample_size = min(200, len(self.X_train_transformed))
        idx = np.random.RandomState(RANDOM_SEED).choice(
            len(self.X_train_transformed), sample_size, replace=False
        )
        X_sample = self.X_train_transformed[idx]

        shap_values = self.explainer.shap_values(X_sample)

        # Clase Inactivo
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # Mean absolute SHAP value por feature
        mean_abs = np.abs(sv).mean(axis=0)

        importance = {}
        for name, val in zip(self.transformed_feature_names, mean_abs):
            importance[name] = round(float(val), 6)

        # Ordenar por importancia
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "feature_importance": sorted_importance,
            "sample_size": sample_size,
            "method": "mean(|SHAP|)",
        }


# Singleton
_explainer: SHAPExplainer | None = None


def get_explainer() -> SHAPExplainer:
    """Obtiene la instancia singleton del explainer."""
    global _explainer
    if _explainer is None:
        _explainer = SHAPExplainer()
    return _explainer
