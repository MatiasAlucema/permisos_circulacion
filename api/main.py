"""
main.py -- API REST para servir predicciones del modelo de permisos de circulacion.

Endpoints:
    GET  /health               -- Health check + info del modelo
    POST /predict              -- Prediccion individual
    POST /predict/batch        -- Prediccion en lote (max 100)
    POST /predict/explain      -- Prediccion + explicacion LLM
    GET  /monitoring/drift     -- Reporte de data drift
    GET  /monitoring/status    -- Estado del monitor (requests acumulados)

Ejecutar:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Agregar root del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    ExplainedPredictionResponse,
    HealthResponse,
    PermitPredictionRequest,
    PredictionResponse,
)
from src.llm_explainer import explain_prediction
from src.monitoring import DriftMonitor, get_monitor
from src.predict import PermitPredictor, get_predictor

# ── Lifespan: carga el modelo y monitor al iniciar ────────

predictor: PermitPredictor | None = None
monitor: DriftMonitor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y monitor al iniciar, limpia al cerrar."""
    global predictor, monitor
    print("[LOADING] Cargando modelo...")
    try:
        predictor = get_predictor()
        info = predictor.get_model_info()
        print(f"[OK] Modelo cargado: {info['model_type']}")
        print(f"  Entrenado: {info['trained_at']}")
        print(f"  F1 test: {info['test_metrics']['f1_score']}")
    except Exception as e:
        print(f"[ERROR] Error cargando modelo: {e}")
        predictor = None

    print("[LOADING] Iniciando monitor de drift...")
    try:
        monitor = get_monitor()
        ref_count = len(monitor.reference_data) if monitor.reference_data is not None else 0
        print(f"[OK] Monitor activo (referencia: {ref_count} registros)")
    except Exception as e:
        print(f"[ERROR] Error iniciando monitor: {e}")
        monitor = None

    yield
    print("[STOP] Cerrando API...")


# ── App FastAPI ───────────────────────────────────────────

app = FastAPI(
    title="Permisos de Circulacion -- API de Prediccion",
    description=(
        "API REST para predecir si un permiso de circulacion sera **Activo** o **Inactivo**.\n\n"
        "Incluye:\n"
        "- `/predict/explain` -- Explicaciones con IA (Gemini/Claude)\n"
        "- `/monitoring/drift` -- Deteccion de data drift con Evidently AI\n"
    ),
    version="3.0.0",
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints: Prediccion ─────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health_check():
    """Verifica el estado de la API y del modelo cargado."""
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False)

    info = predictor.get_model_info()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_type=info["model_type"],
        trained_at=info["trained_at"],
        test_metrics=info["test_metrics"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediccion"])
async def predict(request: PermitPredictionRequest):
    """Predice el estado de un permiso de circulacion individual."""
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Modelo no disponible. Ejecute train.py primero."
        )

    try:
        result = predictor.predict_single(request.model_dump())

        # Log para monitoring de drift
        if monitor:
            monitor.log_prediction(request.model_dump())

        return PredictionResponse(
            prediction=result.prediction,
            probability_inactive=result.probability_inactive,
            probability_active=result.probability_active,
            features_used=result.features_used,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediccion"])
async def predict_batch(request: BatchPredictionRequest):
    """Predice el estado de multiples permisos (max 100 por request)."""
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Modelo no disponible. Ejecute train.py primero."
        )

    try:
        results = predictor.predict_batch([p.model_dump() for p in request.permits])

        # Log para monitoring de drift
        if monitor:
            for p in request.permits:
                monitor.log_prediction(p.model_dump())

        predictions = [
            PredictionResponse(
                prediction=r.prediction,
                probability_inactive=r.probability_inactive,
                probability_active=r.probability_active,
                features_used=r.features_used,
            )
            for r in results
        ]
        return BatchPredictionResponse(predictions=predictions, total=len(predictions))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion batch: {str(e)}")


@app.post("/predict/explain", response_model=ExplainedPredictionResponse, tags=["IA"])
async def predict_with_explanation(request: PermitPredictionRequest):
    """Predice el estado de un permiso y genera una explicacion con IA.

    Flujo:
    1. El modelo ML predice Activo/Inactivo con probabilidades
    2. Un LLM (Gemini o Claude) analiza la prediccion + features y genera
       una explicacion en lenguaje natural

    **Requiere**: Variable de entorno `GEMINI_API_KEY` o `ANTHROPIC_API_KEY` configurada.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Modelo no disponible. Ejecute train.py primero."
        )

    try:
        result = predictor.predict_single(request.model_dump())

        # Log para monitoring
        if monitor:
            monitor.log_prediction(request.model_dump())

        explanation = await explain_prediction(
            prediction=result.prediction,
            probability_inactive=result.probability_inactive,
            probability_active=result.probability_active,
            features=request.model_dump(),
        )

        return ExplainedPredictionResponse(
            prediction=result.prediction,
            probability_inactive=result.probability_inactive,
            probability_active=result.probability_active,
            features_used=result.features_used,
            explanation=explanation.explanation,
            llm_model=explanation.model_used,
            llm_tokens_used=explanation.tokens_used,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en prediccion/explicacion: {str(e)}")


# ── Endpoints: Monitoring ─────────────────────────────────


@app.get("/monitoring/status", tags=["Monitoring"])
async def monitoring_status():
    """Estado del monitor de drift: requests acumulados y datos de referencia."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor no disponible.")

    ref_count = len(monitor.reference_data) if monitor.reference_data is not None else 0
    return {
        "production_requests_logged": monitor.get_production_count(),
        "reference_samples": ref_count,
        "min_samples_for_drift": monitor.min_samples,
        "ready_for_drift_check": monitor.get_production_count() >= monitor.min_samples,
    }


@app.get("/monitoring/drift", tags=["Monitoring"])
async def check_drift():
    """Ejecuta deteccion de data drift comparando requests vs datos de entrenamiento.

    Compara las distribuciones de features de los requests recibidos contra
    el dataset de entrenamiento usando tests estadisticos (K-S para numericas,
    chi-square para categoricas).

    **Requiere** al menos 30 requests acumulados para calcular drift.
    """
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor no disponible.")

    result = monitor.check_drift()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.post("/monitoring/clear", tags=["Monitoring"])
async def clear_monitoring():
    """Limpia los datos de produccion acumulados (reset del monitor)."""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor no disponible.")

    cleared = monitor.clear_production_data()
    return {
        "cleared_records": cleared,
        "message": f"Se eliminaron {cleared} registros de produccion.",
    }


# ── Endpoints: Sistema ────────────────────────────────────


@app.get("/model/info", tags=["Sistema"])
async def model_info():
    """Retorna metadata completa del modelo."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    return predictor.get_model_info()
