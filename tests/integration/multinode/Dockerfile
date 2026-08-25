# Hub container for multi-node scale load test.
# Linux-runnable (FastAPI); MLX stays on the host — this container only runs
# the Hub, pointing FMH_MLX_URL at the real MLX (host) or a mock backend.
#
# No apt/build-essential: cryptography etc. ship prebuilt manylinux wheels for
# cp312, so pip pulls binaries and skips compilation. (apt downloads off
# deb.debian.org were pathologically slow in OrbStack here — a 45min stall on
# gcc. Wheels avoid the toolchain entirely.)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

WORKDIR /app

# Install the Hub editable. Source is COPY'd (not bind-mounted) for a
# self-contained image. Auth off so the load driver needs no keys (test scope
# is routing/throughput, not auth — auth is covered by the 860 unit suite).
COPY pyproject.toml README.md /app/
COPY fusion_model_hub /app/fusion_model_hub
RUN pip install --upgrade pip && pip install -e .

ENV FMH_DATA_DIR=/data \
    MODEL_HUB_AUTH_ENABLED=false \
    FMH_EXPOSE_METRICS=true

EXPOSE 11444

# serve subcommand (per CLAUDE.md: NOT bare invocation).
CMD ["fusion-model-hub", "serve", "--host", "0.0.0.0", "--port", "11444"]
