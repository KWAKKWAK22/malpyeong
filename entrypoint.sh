#!/usr/bin/env bash
set -euo pipefail

python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 127.0.0.1 --port 8001 \
  --served-model-name "${MODEL_NAME}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 &

exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
