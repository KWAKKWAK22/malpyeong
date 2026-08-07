#!/usr/bin/env bash
# RunPod Pod 안에서 컨테이너 없이 코드만 띄우는 개발용 스크립트.
# Pod에서는 nginx가 8001을 선점하므로 vLLM을 18001로 올린다.
# (컨테이너 안에서는 entrypoint.sh가 8001을 그대로 쓴다.)
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kwriting-scorer}"
export VLLM_PORT="${VLLM_PORT:-18001}"
export APP_PORT="${APP_PORT:-8000}"
export VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBUG_RAW="${DEBUG_RAW:-1}"      # raw 예측값 저장 (격자 탐색용)
export no_proxy='*'; export NO_PROXY='*'   # Pod 웹터미널 프록시 우회

mkdir -p logs
echo "[1/3] vLLM 기동 (port ${VLLM_PORT}) — 첫 실행은 모델 다운로드로 5~10분"
python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 127.0.0.1 --port "${VLLM_PORT}" \
  --served-model-name "${MODEL_NAME}" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 > logs/vllm.log 2>&1 &
VLLM_PID=$!

echo "[2/3] vLLM 준비 대기"
for i in $(seq 1 180); do
  if curl -s --noproxy '*' "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null; then
    echo "  vLLM ready (${i}0s)"; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "  ! vLLM 죽음. logs/vllm.log 확인"; tail -30 logs/vllm.log; exit 1
  fi
  sleep 10
done

echo "[3/3] 래퍼 기동 (port ${APP_PORT})"
exec python3 -m uvicorn server:app --host 0.0.0.0 --port "${APP_PORT}" --workers 1
