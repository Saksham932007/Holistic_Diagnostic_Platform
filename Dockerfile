# Medical Diagnostic Platform Docker Configuration
# Multi-stage build for optimized production deployment
# with security hardening and minimal attack surface

# Stage 1: Base Python environment with system dependencies
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04 AS python-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopencv-dev \
    libfreetype6-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r meduser && useradd -r -g meduser meduser

# Stage 2: Python dependencies installation
FROM python-base AS deps

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# Stage 3: Development environment (for testing)
FROM deps AS development

# Install development dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements-dev.txt

# Copy application code
WORKDIR /app
COPY . .

# Change ownership to non-root user
RUN chown -R meduser:meduser /app

# Switch to non-root user
USER meduser

# Expose port
EXPOSE 8000

# Development command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production environment
FROM python-base AS production

# Install only production Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# Create application directory
WORKDIR /app

# Copy application code (excluding development files)
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/checkpoints /app/cache && \
    chmod 755 /app/data /app/logs /app/checkpoints /app/cache

# Copy startup script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Security hardening
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Change ownership to non-root user
RUN chown -R meduser:meduser /app /entrypoint.sh

# Switch to non-root user
USER meduser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 5: Inference-only lightweight image
FROM python-base AS inference

# Install minimal dependencies for inference
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    numpy \
    nibabel \
    pillow \
    fastapi \
    uvicorn \
    pydantic

# Copy only inference-related code
WORKDIR /app
COPY src/models/ ./src/models/
COPY src/preprocessing/ ./src/preprocessing/
COPY src/inference/ ./src/inference/
COPY src/api/main.py ./src/api/main.py
COPY src/api/security.py ./src/api/security.py

# Create minimal configuration
RUN echo '{"model_dir": "/app/models", "cache_dir": "/app/cache"}' > /app/config.json

# Change ownership
RUN chown -R meduser:meduser /app
USER meduser

# Expose port
EXPOSE 8000

# Inference-only command
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]