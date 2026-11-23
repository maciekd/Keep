#!/bin/bash
# Starts oMLX (github.com/jundot/omlx) serving Qwen 3.8 27B (MLX 4-bit) for Keep's `local_backend: openai`,
# OpenAI-style on 127.0.0.1:8000 (`openai_compatible.base_url`). The server's own model id is the
# folder name under ~/.omlx/models, Qwen3.8-27B-4bit — `openai_compatible.model` must name it, since a
# model whose name says qwen is what makes Keep write Qwen's call syntax. Runs in the foreground,
# its output also in log/omlx.log; `--stop` stops a running one. Mac only.
#
# Memory: oMLX's guard refuses to load a model that does not fit under a ceiling it derives from
# what other apps hold, and on a 32 GB Mac with an IDE and a browser open both its balanced and
# aggressive tiers leave ~12 GB for the 15.7 GB model. A fixed ceiling (MEMORY_GB, default 22 —
# the weights plus ~6 GB of KV cache) loads it and lets macOS compress the other apps. Measured on
# an M1 Max, 32 GB (2026-11-23): ~17 tok/s decode, ~80 tok/s prefill, a cached 11.7k-token prefix 1 s.
# The prefix cache is oMLX's own, on disk under ~/.omlx/cache, and survives a restart.
#
# Install: brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx
# The model (15 GB) is downloaded on the first start when the folder is missing.
#
# Overrides: MODEL_REPO, MODEL_DIR, HOST (0.0.0.0 for a Keep elsewhere on the LAN), PORT, MEMORY_GB.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p log

MODEL_REPO="${MODEL_REPO:-mlx-community/Qwen3.8-27B-4bit}"
MODEL_DIR="${MODEL_DIR:-$HOME/.omlx/models}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MEMORY_GB="${MEMORY_GB:-22}"

stop_omlx() {
    if ! pgrep -x omlx-server > /dev/null; then
        echo "No oMLX running."
        return 0
    fi
    echo "Stopping oMLX..."
    pkill -x omlx-server
    for _ in $(seq 1 50); do
        pgrep -x omlx-server > /dev/null || { echo "oMLX stopped."; return 0; }
        sleep 0.2
    done
    echo "Force-killing oMLX..."
    pkill -9 -x omlx-server || true
}

if [ "${1:-}" = "--stop" ]; then
    stop_omlx
    exit 0
fi

command -v omlx > /dev/null || {
    echo "oMLX is not installed: brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx"
    exit 1
}

model_path="$MODEL_DIR/${MODEL_REPO##*/}"
if [ ! -f "$model_path/config.json" ]; then
    echo "Downloading $MODEL_REPO → $model_path"
    hf=".venv/bin/hf"
    [ -x "$hf" ] || hf="$(command -v hf)" || { echo "no hf CLI: pip install huggingface_hub"; exit 1; }
    "$hf" download "$MODEL_REPO" --local-dir "$model_path"
fi

stop_omlx

echo "Starting oMLX on $HOST:$PORT (model id ${model_path##*/}, memory ceiling ${MEMORY_GB} GB) → log/omlx.log"
omlx serve --model-dir "$MODEL_DIR" --host "$HOST" --port "$PORT" --memory-guard-gb "$MEMORY_GB" 2>&1 | tee log/omlx.log
