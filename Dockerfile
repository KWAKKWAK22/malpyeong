FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
ENV SERVED_MODEL_NAME=kwriting-scorer
ENV VLLM_URL=http://127.0.0.1:8001
ENV HF_HOME=/models
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3 python3-pip curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --break-system-packages \
    "vllm==0.11.0" "transformers==4.56.2" hf_transfer httpx uvicorn fastapi \
    --extra-index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
