# ── Stage 1: Build ────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalar dependencias del sistema para scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ── Stage 2: Runtime ─────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /install /usr/local

# Copiar código del proyecto
COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY data/processed/ data/processed/
COPY models/ models/

# Puerto de la API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Ejecutar la API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
