# Multi-stage production Dockerfile for B2B SaaS Churn Early-Warning Microservice

# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies required for compiling extensions (LightGBM, CFFI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Minimal Runtime
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install minimal runtime libraries (libgomp for LightGBM OpenMP parallelization, curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Create non-privileged system user for container security hardening
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 -s /sbin/nologin appuser

# Copy application code, trained model artifacts, and project configs
COPY src/ /app/src/
COPY models/ /app/models/
COPY monitoring/ /app/monitoring/
COPY pyproject.toml /app/

# Ensure monitoring reports directory exists and is writable
RUN mkdir -p /app/monitoring/reports

# Install the application in editable/packaged mode into the virtualenv
RUN pip install --no-deps -e .

# Transfer file ownership to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Docker healthcheck querying the /health probe
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "30"]
