FROM vllm/vllm-openai:v0.11.0

ENV MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
ENV SERVED_MODEL_NAME=kwriting-scorer
ENV VLLM_URL=http://127.0.0.1:8001
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV DEBIAN_FRONTEND=noninteractive

# 공식 이미지에 vllm/torch/transformers/hf_transfer/fastapi/uvicorn 포함.
# 래퍼가 추가로 쓰는 것만 설치.
RUN pip install --no-cache-dir httpx fastapi uvicorn

WORKDIR /app
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
# 공식 이미지의 ENTRYPOINT를 반드시 덮어쓴다 (기본값이 vllm serve).
ENTRYPOINT ["/app/entrypoint.sh"]
