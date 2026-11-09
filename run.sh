#!/usr/bin/env bash
# ============================================================
# Video Swap Pipeline — Local Development Runner
# Usage: ./run.sh [api|worker|all|flower]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "✅ Loaded .env"
fi

# Create directories
mkdir -p "${VSWAP_UPLOAD_DIR:-/tmp/vswap/uploads}"
mkdir -p "${VSWAP_PROCESSED_DIR:-/tmp/vswap/processed}"
mkdir -p "${VSWAP_OUTPUT_DIR:-/tmp/vswap/output}"
mkdir -p logs

MODE="${1:-all}"

start_redis() {
    if ! pgrep -x redis-server > /dev/null 2>&1; then
        echo "🔴 Starting Redis..."
        redis-server --daemonize yes --loglevel warning
        sleep 1
        echo "✅ Redis running on port 6379"
    else
        echo "✅ Redis already running"
    fi
}

start_worker() {
    echo "⚙️  Starting Celery worker..."
    celery -A workers.celery_app worker \
        --loglevel=info \
        --concurrency=2 \
        -Q chunks,stitch \
        --logfile=logs/worker.log \
        --pidfile=logs/worker.pid &
    echo "✅ Worker started (PID: $!)"
}

start_api() {
    echo "🚀 Starting FastAPI server..."
    uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload \
        --log-level info &
    echo "✅ API server at http://0.0.0.0:8000"
    echo "📖 Docs at http://0.0.0.0:8000/docs"
}

start_flower() {
    echo "🌸 Starting Flower monitor..."
    celery -A workers.celery_app flower \
        --port=5555 \
        --logfile=logs/flower.log &
    echo "✅ Flower at http://0.0.0.0:5555"
}

case "$MODE" in
    api)
        start_redis
        start_api
        ;;
    worker)
        start_redis
        start_worker
        ;;
    flower)
        start_redis
        start_flower
        ;;
    all)
        start_redis
        start_worker
        start_api
        echo ""
        echo "=========================================="
        echo "🎭 Video Swap Pipeline is running!"
        echo "   API:    http://localhost:8000"
        echo "   Docs:   http://localhost:8000/docs"
        echo "   Health: http://localhost:8000/health"
        echo "=========================================="
        ;;
    stop)
        echo "Stopping all services..."
        pkill -f "celery.*workers.celery_app" 2>/dev/null || true
        pkill -f "uvicorn.*main:app" 2>/dev/null || true
        redis-cli shutdown 2>/dev/null || true
        echo "✅ All services stopped"
        exit 0
        ;;
    *)
        echo "Usage: $0 [api|worker|flower|all|stop]"
        exit 1
        ;;
esac

# Wait for all background processes
wait
