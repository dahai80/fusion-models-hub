FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY fusion_model_hub/ fusion_model_hub/
COPY alembic/ alembic/
COPY alembic.ini ./

RUN pip install --no-cache-dir .

ENV FMH_DATA_DIR=/data
ENV FMH_HOST=127.0.0.1
ENV FMH_PORT=11444

RUN mkdir -p /data

EXPOSE 11444

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11444/api/v1/system/health || exit 1

CMD ["fusion-model-hub", "--host", "127.0.0.1", "--port", "11444"]
