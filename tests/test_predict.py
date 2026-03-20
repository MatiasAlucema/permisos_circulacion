"""Tests para el modulo de prediccion (src/predict.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import PredictionResult, get_predictor

# --- Fixtures ---


@pytest.fixture(scope="module")
def predictor():
    """Carga el predictor una vez para todos los tests."""
    return get_predictor()


@pytest.fixture
def sample_permit():
    """Datos de ejemplo de un permiso."""
    return {
        "tipo_vehiculo": "Camion",
        "duracion_dias": 20.0,
        "zona_circulacion": "Zona A",
        "monto_pagado": 85000.0,
        "renovacion": False,
        "infracciones_previas": 3,
        "fecha_emision": "2025-03-15",
    }


@pytest.fixture
def low_risk_permit():
    """Permiso de bajo riesgo."""
    return {
        "tipo_vehiculo": "Coche",
        "duracion_dias": 10.0,
        "zona_circulacion": "Zona C",
        "monto_pagado": 150000.0,
        "renovacion": True,
        "infracciones_previas": 0,
        "fecha_emision": "2025-06-01",
    }


# --- Tests ---


class TestPredictor:
    """Tests del predictor de permisos."""

    def test_predict_returns_result(self, predictor, sample_permit):
        """Prediccion retorna un PredictionResult valido."""
        result = predictor.predict_single(sample_permit)
        assert isinstance(result, PredictionResult)

    def test_prediction_is_valid_label(self, predictor, sample_permit):
        """La prediccion es 'Activo' o 'Inactivo'."""
        result = predictor.predict_single(sample_permit)
        assert result.prediction in ("Activo", "Inactivo")

    def test_probabilities_sum_to_one(self, predictor, sample_permit):
        """Las probabilidades suman ~1.0."""
        result = predictor.predict_single(sample_permit)
        total = result.probability_active + result.probability_inactive
        assert abs(total - 1.0) < 0.01

    def test_probabilities_are_valid(self, predictor, sample_permit):
        """Las probabilidades estan entre 0 y 1."""
        result = predictor.predict_single(sample_permit)
        assert 0 <= result.probability_active <= 1
        assert 0 <= result.probability_inactive <= 1

    def test_high_risk_permit_tends_inactive(self, predictor):
        """Un permiso de alto riesgo deberia tender a Inactivo."""
        high_risk = {
            "tipo_vehiculo": "Monopatin",
            "duracion_dias": 28.0,
            "zona_circulacion": "Zona A",
            "monto_pagado": 3000.0,
            "renovacion": False,
            "infracciones_previas": 5,
            "fecha_emision": "2025-01-11",  # sabado
        }
        result = predictor.predict_single(high_risk)
        # Con tantos factores de riesgo, probabilidad de inactivo deberia ser alta
        assert result.probability_inactive > 0.4

    def test_low_risk_permit_tends_active(self, predictor, low_risk_permit):
        """Un permiso de bajo riesgo deberia tender a Activo."""
        result = predictor.predict_single(low_risk_permit)
        assert result.probability_active > 0.5

    def test_features_used_contains_expected_keys(self, predictor, sample_permit):
        """Las features procesadas incluyen las columnas esperadas."""
        result = predictor.predict_single(sample_permit)
        expected = {
            "tipo_vehiculo",
            "duracion_dias",
            "zona_circulacion",
            "monto_pagado",
            "infracciones_previas",
            "mes",
            "dia_semana",
            "es_fin_semana",
            "dias_antiguedad",
        }
        assert expected.issubset(set(result.features_used.keys()))

    def test_missing_feature_raises_error(self, predictor):
        """Features faltantes generan ValueError."""
        incomplete = {"tipo_vehiculo": "Coche"}
        with pytest.raises((ValueError, KeyError)):
            predictor.predict_single(incomplete)

    def test_batch_prediction(self, predictor, sample_permit, low_risk_permit):
        """Prediccion batch retorna lista correcta."""
        results = predictor.predict_batch([sample_permit, low_risk_permit])
        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_model_info_has_required_fields(self, predictor):
        """La metadata del modelo tiene los campos requeridos."""
        info = predictor.get_model_info()
        assert "model_type" in info
        assert "trained_at" in info
        assert "test_metrics" in info
        assert "features" in info

    def test_model_metrics_are_reasonable(self, predictor):
        """Las metricas del modelo son razonables (no triviales)."""
        info = predictor.get_model_info()
        metrics = info["test_metrics"]
        assert metrics["accuracy"] > 0.5  # Mejor que random
        assert metrics["f1_score"] > 0.3  # F1 razonable
        assert metrics["roc_auc"] > 0.5  # Mejor que random
