"""
schemas.py — Schemas Pydantic v2 para validación de request/response de la API.
"""

from typing import Literal

from pydantic import BaseModel, Field

# ── Request ──────────────────────────────────────────────


class PermitPredictionRequest(BaseModel):
    """Schema de entrada para predecir el estado de un permiso."""

    tipo_vehiculo: Literal["Coche", "Moto", "Camion", "Furgoneta", "Bicicleta", "Monopatin"] = (
        Field(..., description="Tipo de vehículo", examples=["Camion"])
    )
    duracion_dias: float = Field(
        ..., gt=0, le=365, description="Duración del permiso en días", examples=[15.0]
    )
    zona_circulacion: Literal["Zona A", "Zona B", "Zona C", "Zona D"] = Field(
        ..., description="Zona de circulación asignada", examples=["Zona A"]
    )
    monto_pagado: float = Field(
        ..., gt=0, description="Monto pagado por el permiso (CLP)", examples=[85000.0]
    )
    renovacion: bool = Field(
        ..., description="True si es renovación, False si es primera emisión", examples=[False]
    )
    infracciones_previas: int = Field(
        ..., ge=0, le=10, description="Número de infracciones previas del titular", examples=[2]
    )
    fecha_emision: str = Field(
        ..., description="Fecha de emisión (YYYY-MM-DD)", examples=["2025-03-15"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tipo_vehiculo": "Camion",
                    "duracion_dias": 20,
                    "zona_circulacion": "Zona A",
                    "monto_pagado": 85000,
                    "renovacion": False,
                    "infracciones_previas": 2,
                    "fecha_emision": "2025-03-15",
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Schema para predicciones en lote."""

    permits: list[PermitPredictionRequest] = Field(
        ..., min_length=1, max_length=100, description="Lista de permisos a predecir (máx 100)"
    )


# ── Response ─────────────────────────────────────────────


class PredictionResponse(BaseModel):
    """Schema de respuesta para una predicción."""

    prediction: Literal["Activo", "Inactivo"] = Field(
        ..., description="Estado predicho del permiso"
    )
    probability_inactive: float = Field(..., ge=0, le=1, description="Probabilidad de ser Inactivo")
    probability_active: float = Field(..., ge=0, le=1, description="Probabilidad de ser Activo")
    features_used: dict = Field(..., description="Features procesadas utilizadas por el modelo")


class BatchPredictionResponse(BaseModel):
    """Schema de respuesta para predicciones en lote."""

    predictions: list[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Schema de respuesta del health check."""

    status: str = "ok"
    model_loaded: bool
    model_type: str | None = None
    trained_at: str | None = None
    test_metrics: dict | None = None


class ExplainedPredictionResponse(BaseModel):
    """Schema de respuesta para prediccion + explicacion LLM."""

    prediction: Literal["Activo", "Inactivo"] = Field(
        ..., description="Estado predicho del permiso"
    )
    probability_inactive: float = Field(..., ge=0, le=1, description="Probabilidad de ser Inactivo")
    probability_active: float = Field(..., ge=0, le=1, description="Probabilidad de ser Activo")
    features_used: dict = Field(..., description="Features procesadas utilizadas por el modelo")
    explanation: str = Field(..., description="Explicacion en lenguaje natural generada por Claude")
    llm_model: str = Field(..., description="Modelo de LLM utilizado para la explicacion")
    llm_tokens_used: int = Field(..., ge=0, description="Tokens consumidos por la explicacion")


class ErrorResponse(BaseModel):
    """Schema de error."""

    detail: str
