FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-predictor.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-predictor.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r predictor && useradd -r -g predictor predictor

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY predictor-entrypoint.py ./

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs && \
    chown -R predictor:predictor /app

USER predictor

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_CACHE_DIR=/app/models \
    LOG_LEVEL=INFO \
    REDIS_CLUSTER_HOST=redis-cluster \
    REDIS_CLUSTER_PORT=6379 \
    PREDICTOR_BATCH_SIZE=32 \
    PREDICTOR_QUEUE_SIZE=1000 \
    PREDICTOR_WORKERS=4 \
    PRECOMPUTE_INTERVAL=300 \
    SIMILARITY_THRESHOLD=0.85 \
    MODEL_UPDATE_INTERVAL=3600

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health', timeout=5)" || exit 1

EXPOSE 8002

CMD ["python", "predictor-entrypoint.py"]