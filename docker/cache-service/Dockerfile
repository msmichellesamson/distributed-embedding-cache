FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements/cache-service.txt ./requirements.txt

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim as runtime

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/* && \
        groupadd -r cache && \
        useradd -r -g cache -s /bin/false cache

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models && \
    chown -R cache:cache /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/start-cache-service.sh ./scripts/

# Set permissions
RUN chmod +x scripts/start-cache-service.sh && \
    chown -R cache:cache /app

USER cache

# Environment variables
ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_SERVICE_PORT=8080 \
    GRPC_SERVICE_PORT=9090 \
    REDIS_CLUSTER_NODES="redis-0:6379,redis-1:6379,redis-2:6379" \
    LOG_LEVEL=INFO \
    METRICS_PORT=9091 \
    MODEL_PATH="/app/models" \
    PRECOMPUTE_BATCH_SIZE=256 \
    MAX_CACHE_SIZE_MB=1024

EXPOSE 8080 9090 9091

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["./scripts/start-cache-service.sh"]