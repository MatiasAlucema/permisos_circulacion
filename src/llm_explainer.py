"""
llm_explainer.py -- Genera explicaciones en lenguaje natural de las predicciones
usando un LLM (soporta Gemini y Claude).

Recibe la prediccion del modelo ML + features del permiso y produce:
- Explicacion clara de por que el modelo predice Activo/Inactivo
- Factores de riesgo identificados
- Recomendacion de accion para el funcionario

Configuracion via .env:
    LLM_PROVIDER=gemini          # "gemini" o "anthropic"
    GEMINI_API_KEY=...           # Si provider es gemini
    ANTHROPIC_API_KEY=...        # Si provider es anthropic
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --- Config ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# System prompt con contexto del dominio
SYSTEM_PROMPT = """Eres un asistente experto en gestion de permisos de circulacion municipales.

Tu trabajo es analizar predicciones de un modelo de machine learning que clasifica permisos
como "Activo" o "Inactivo" (riesgo de irregularidad/expiracion).

Contexto del modelo:
- Usa RandomForest + SMOTE entrenado sobre datos historicos de permisos
- Features principales: tipo_vehiculo, zona_circulacion, infracciones_previas, monto_pagado,
  duracion_dias, renovacion, antiguedad del permiso, dia de la semana
- Los factores mas predictivos son: infracciones_previas, dias_antiguedad, monto_pagado

Reglas de negocio conocidas:
- Camiones en Zona A tienen restricciones especiales (zona regulada)
- Monopatines tienen regulacion reciente y mayor tasa de inactividad
- Permisos emitidos en fin de semana tienen mayor riesgo
- Mas infracciones previas = mayor probabilidad de inactividad
- Montos muy bajos respecto al tipo de vehiculo son sospechosos
- Las renovaciones tienden a ser mas estables que las primeras emisiones

Responde SIEMPRE en espanol y con formato estructurado."""


def _build_user_prompt(
    prediction: str, probability_inactive: float, probability_active: float, features: dict
) -> str:
    """Construye el prompt del usuario con los datos de la prediccion."""
    max_prob = max(probability_inactive, probability_active)
    if max_prob >= 0.75:
        confidence = "alta"
    elif max_prob >= 0.60:
        confidence = "moderada"
    else:
        confidence = "baja"

    monto = features.get("monto_pagado", 0)
    monto_str = f"${monto:,.0f}" if isinstance(monto, (int, float)) else str(monto)

    return f"""Analiza esta prediccion de un permiso de circulacion:

## Prediccion del modelo
- **Estado predicho**: {prediction}
- **Probabilidad Inactivo**: {probability_inactive:.1%}
- **Probabilidad Activo**: {probability_active:.1%}
- **Confianza**: {confidence}

## Datos del permiso
- Tipo de vehiculo: {features.get("tipo_vehiculo", "N/A")}
- Zona de circulacion: {features.get("zona_circulacion", "N/A")}
- Duracion del permiso: {features.get("duracion_dias", "N/A")} dias
- Monto pagado: {monto_str} CLP
- Renovacion: {"Si" if features.get("renovacion") else "No"}
- Infracciones previas: {features.get("infracciones_previas", "N/A")}
- Emitido en fin de semana: {"Si" if features.get("es_fin_semana") else "No"}
- Antiguedad: {features.get("dias_antiguedad", "N/A")} dias

Responde con exactamente este formato:

### Explicacion
[2-3 oraciones explicando por que el modelo predice este estado, mencionando los factores clave]

### Factores de riesgo
[Lista con los factores que mas influyen en esta prediccion, ordenados por importancia]

### Recomendacion
[1-2 oraciones con la accion recomendada para el funcionario municipal]

### Nivel de confianza
[Una oracion sobre que tan confiable es esta prediccion y por que]"""


@dataclass
class ExplanationResult:
    """Resultado de la explicacion generada por el LLM."""

    explanation: str
    model_used: str
    tokens_used: int


async def _explain_with_gemini(user_prompt: str) -> ExplanationResult:
    """Genera explicacion usando Google Gemini."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY no configurada. Crea un archivo .env con: GEMINI_API_KEY=..."
        )

    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.0-flash"

    response = client.models.generate_content(
        model=model_name,
        contents=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
    )

    tokens_used = 0
    if response.usage_metadata:
        tokens_used = (response.usage_metadata.prompt_token_count or 0) + (
            response.usage_metadata.candidates_token_count or 0
        )

    return ExplanationResult(
        explanation=response.text,
        model_used=model_name,
        tokens_used=tokens_used,
    )


async def _explain_with_anthropic(user_prompt: str) -> ExplanationResult:
    """Genera explicacion usando Anthropic Claude."""
    from anthropic import AsyncAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY no configurada. "
            "Crea un archivo .env con: ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = AsyncAnthropic(api_key=api_key)
    model_name = "claude-sonnet-4-20250514"

    response = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    total_tokens = response.usage.input_tokens + response.usage.output_tokens

    return ExplanationResult(
        explanation=response.content[0].text,
        model_used=model_name,
        tokens_used=total_tokens,
    )


async def explain_prediction(
    prediction: str,
    probability_inactive: float,
    probability_active: float,
    features: dict,
) -> ExplanationResult:
    """Genera una explicacion en lenguaje natural de la prediccion.

    Soporta multiples providers de LLM configurables via .env:
    - LLM_PROVIDER=gemini  (default) -> usa Google Gemini
    - LLM_PROVIDER=anthropic         -> usa Claude (Anthropic)

    Args:
        prediction: "Activo" o "Inactivo"
        probability_inactive: Probabilidad de ser inactivo (0-1)
        probability_active: Probabilidad de ser activo (0-1)
        features: Dict con las features del permiso

    Returns:
        ExplanationResult con la explicacion, modelo usado y tokens consumidos.
    """
    user_prompt = _build_user_prompt(prediction, probability_inactive, probability_active, features)

    if LLM_PROVIDER == "anthropic":
        return await _explain_with_anthropic(user_prompt)
    elif LLM_PROVIDER == "gemini":
        return await _explain_with_gemini(user_prompt)
    else:
        raise ValueError(
            f"LLM_PROVIDER no soportado: '{LLM_PROVIDER}'. Usa 'gemini' o 'anthropic'."
        )
