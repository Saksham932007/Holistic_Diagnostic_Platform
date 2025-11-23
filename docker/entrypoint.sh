#!/bin/bash

# Medical Diagnostic Platform Docker Entrypoint Script
# Handles initialization, security, and service startup

set -e

echo "=== Medical Diagnostic Platform Startup ==="
echo "Timestamp: $(date)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check environment variables
check_env() {
    log "Checking environment variables..."
    
    # Required environment variables
    REQUIRED_VARS=(
        "JWT_SECRET_KEY"
        "DATABASE_URL"
    )
    
    # Check if required variables are set
    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var}" ]; then
            log "WARNING: Required environment variable $var is not set"
            case $var in
                "JWT_SECRET_KEY")
                    export JWT_SECRET_KEY="$(openssl rand -hex 32)"
                    log "Generated temporary JWT_SECRET_KEY"
                    ;;
                "DATABASE_URL")
                    export DATABASE_URL="sqlite:///app/data/medical_platform.db"
                    log "Using default SQLite database"
                    ;;
            esac
        else
            log "✓ $var is set"
        fi
    done
    
    # Optional environment variables with defaults
    export REDIS_URL="${REDIS_URL:-redis://redis:6379}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export WORKERS="${WORKERS:-4}"
    export MAX_WORKERS="${MAX_WORKERS:-8}"
    export PORT="${PORT:-8000}"
    export HOST="${HOST:-0.0.0.0}"
    
    log "Environment variables configured"
}

# Function to initialize directories
init_directories() {
    log "Initializing directories..."
    
    DIRECTORIES=(
        "/app/data"
        "/app/logs"
        "/app/checkpoints"
        "/app/cache"
        "/app/uploads"
        "/app/results"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
        
        # Ensure proper permissions
        chmod 755 "$dir"
    done
    
    log "Directories initialized"
}

# Function to wait for dependencies
wait_for_dependencies() {
    log "Waiting for dependencies..."
    
    # Wait for Redis if configured
    if [ "$REDIS_URL" != "redis://redis:6379" ] || [ "$NODE_ENV" = "production" ]; then
        log "Checking Redis connection..."
        
        # Extract host and port from Redis URL
        REDIS_HOST=$(echo $REDIS_URL | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
        REDIS_PORT=$(echo $REDIS_URL | sed -n 's/.*:\([0-9]*\).*/\1/p')
        
        # Default port if not specified
        REDIS_PORT=${REDIS_PORT:-6379}
        
        # Wait for Redis to be available
        timeout=30
        while ! nc -z "$REDIS_HOST" "$REDIS_PORT"; do
            log "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
            sleep 2
            timeout=$((timeout - 2))
            
            if [ $timeout -le 0 ]; then
                log "WARNING: Redis connection timeout, continuing without Redis"
                export REDIS_URL=""
                break
            fi
        done
        
        if [ -n "$REDIS_URL" ]; then
            log "✓ Redis connection established"
        fi
    fi
    
    log "Dependencies check completed"
}

# Function to run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Check if database migration script exists
    if [ -f "/app/scripts/migrate.py" ]; then
        python /app/scripts/migrate.py
        log "Database migrations completed"
    else
        log "No migration script found, skipping..."
    fi
}

# Function to validate models
validate_models() {
    log "Validating model files..."
    
    MODEL_DIR="/app/checkpoints"
    
    if [ -d "$MODEL_DIR" ]; then
        model_count=$(find "$MODEL_DIR" -name "*.pth" -o -name "*.pt" | wc -l)
        log "Found $model_count model files in $MODEL_DIR"
        
        # List model files
        if [ $model_count -gt 0 ]; then
            log "Available models:"
            find "$MODEL_DIR" -name "*.pth" -o -name "*.pt" | while read -r model; do
                log "  - $(basename "$model")"
            done
        else
            log "WARNING: No pre-trained models found. API will start but model loading may fail."
        fi
    else
        log "WARNING: Model directory $MODEL_DIR not found"
    fi
}

# Function to check GPU availability
check_gpu() {
    log "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        if [ $gpu_count -gt 0 ]; then
            log "✓ Found $gpu_count GPU(s) available"
            nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | \
            while IFS=',' read -r name memory_total memory_used; do
                log "  GPU: $name, Memory: ${memory_used}MB / ${memory_total}MB"
            done
        else
            log "⚠ GPU driver available but no GPUs detected"
        fi
    else
        log "⚠ No GPU support detected, running on CPU only"
    fi
}

# Function to set up logging
setup_logging() {
    log "Setting up logging configuration..."
    
    # Create logging configuration
    cat > /app/logging.conf << EOF
[loggers]
keys=root,uvicorn,medical_platform

[handlers]
keys=console,file

[formatters]
keys=detailed,simple

[logger_root]
level=${LOG_LEVEL}
handlers=console,file

[logger_uvicorn]
level=${LOG_LEVEL}
handlers=console,file
qualname=uvicorn
propagate=0

[logger_medical_platform]
level=${LOG_LEVEL}
handlers=console,file
qualname=medical_platform
propagate=0

[handler_console]
class=StreamHandler
level=${LOG_LEVEL}
formatter=simple
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=${LOG_LEVEL}
formatter=detailed
args=('/app/logs/medical_platform.log',)

[formatter_simple]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s

[formatter_detailed]
format=%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s
EOF
    
    log "Logging configuration created"
}

# Function to perform health check
health_check() {
    log "Performing initial health check..."
    
    # Check Python modules
    python -c "
import sys
import torch
import numpy as np
import fastapi
import uvicorn

print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
print(f'FastAPI version: {fastapi.__version__}')
print('All core dependencies loaded successfully')
"
    
    log "Health check completed"
}

# Function to start the application
start_application() {
    log "Starting Medical Diagnostic Platform API..."
    
    # Determine number of workers based on CPU cores
    if [ -z "$WORKERS" ]; then
        CPU_COUNT=$(nproc)
        WORKERS=$((CPU_COUNT * 2 + 1))
        if [ $WORKERS -gt $MAX_WORKERS ]; then
            WORKERS=$MAX_WORKERS
        fi
    fi
    
    log "Starting with $WORKERS workers on $HOST:$PORT"
    
    # Start the application with proper arguments
    exec "$@" --workers "$WORKERS" --host "$HOST" --port "$PORT"
}

# Main execution flow
main() {
    log "=== Initialization Phase ==="
    check_env
    init_directories
    setup_logging
    
    log "=== Dependency Phase ==="
    wait_for_dependencies
    
    log "=== Validation Phase ==="
    validate_models
    check_gpu
    health_check
    
    log "=== Migration Phase ==="
    run_migrations
    
    log "=== Startup Phase ==="
    start_application "$@"
}

# Trap signals for graceful shutdown
trap 'log "Received shutdown signal, stopping services..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"