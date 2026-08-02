FROM vllm/vllm-openai:latest

ENV MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507
ENV SERVED_MODEL_NAME=kwriting-scorer
ENV VLLM_URL=http://127.0.0.1:8001
ENV HF_HOME=/models

RUN pip install --no-cache-dir httpx uvicorn fastapi

WORKDIR /app
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
