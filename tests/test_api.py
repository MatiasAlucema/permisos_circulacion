"""Tests para la API REST (api/main.py)."""

import os
import sys

import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

# --- Fixtures ---

@pytest.fixture
def sample_request():
    """Request de ejemplo."""
    return {
        "tipo_vehiculo": "Camion",
        "duracion_dias": 20,
        "zona_circulacion": "Zona A",
        "monto_pagado": 85000,
        "renovacion": False,
        "infracciones_previas": 3,
        "fecha_emision": "2025-03-15",
    }


# --- Tests ---

class TestHealthEndpoint:
    """Tests del endpoint GET /health."""

    @pytest.mark.anyio
    async def test_health_returns_ok(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    @pytest.mark.anyio
    async def test_health_includes_metrics(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
        data = resp.json()
        assert "test_metrics" in data
        assert data["model_type"] is not None


class TestPredictEndpoint:
    """Tests del endpoint POST /predict."""

    @pytest.mark.anyio
    async def test_predict_success(self, sample_request):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json=sample_request)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in ("Activo", "Inactivo")
        assert 0 <= data["probability_inactive"] <= 1
        assert 0 <= data["probability_active"] <= 1

    @pytest.mark.anyio
    async def test_predict_invalid_vehiculo(self, sample_request):
        """Tipo de vehiculo invalido retorna 422."""
        sample_request["tipo_vehiculo"] = "Avion"
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json=sample_request)
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_predict_negative_duration(self, sample_request):
        """Duracion negativa retorna 422."""
        sample_request["duracion_dias"] = -5
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json=sample_request)
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_predict_missing_field(self):
        """Request incompleto retorna 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json={"tipo_vehiculo": "Coche"})
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_predict_returns_features(self, sample_request):
        """Response incluye las features procesadas."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict", json=sample_request)
        data = resp.json()
        assert "features_used" in data
        assert "infracciones_previas" in data["features_used"]


class TestBatchEndpoint:
    """Tests del endpoint POST /predict/batch."""

    @pytest.mark.anyio
    async def test_batch_predict(self, sample_request):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict/batch", json={"permits": [sample_request, sample_request]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    @pytest.mark.anyio
    async def test_batch_empty_list(self):
        """Lista vacia retorna 422."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/predict/batch", json={"permits": []})
        assert resp.status_code == 422


class TestMonitoringEndpoints:
    """Tests de los endpoints de monitoring."""

    @pytest.mark.anyio
    async def test_monitoring_status(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/monitoring/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "production_requests_logged" in data
        assert "reference_samples" in data

    @pytest.mark.anyio
    async def test_model_info(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_type" in data
        assert "features" in data
