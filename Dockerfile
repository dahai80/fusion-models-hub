FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# P1-21: build context MUST be the monorepo root (not this sub-project) so the
# shared, pinned requirements.lock is copied in. Installing the editable package
# alone would resolve latest-matching deps, drifting from the lock that keeps
# every node identical. Sync the lock first, then install the project on top.
COPY requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml README.md ./
COPY fusion_model_hub/ fusion_model_hub/
COPY alembic/ alembic/
COPY alembic.ini ./

RUN pip install --no-cache-dir --no-deps .

ENV FMH_DATA_DIR=/data
# P1-20: bind 0.0.0.0 so the API is reachable outside the container; 127.0.0.1
# is loopback-only and made the service invisible to anything off the container
# netns. The `serve` subcommand is required — bare `fusion-model-hub` is not a
# valid invocation (per CLAUDE.md: "subcommand serve, NOT bare invocation").
ENV FMH_HOST=0.0.0.0
ENV FMH_PORT=11444

RUN mkdir -p /data

EXPOSE 11444

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11444/api/v1/system/health || exit 1

# P1-20: `serve` subcommand is required (bare invocation is invalid). Omit
# --host/--port so they resolve from FMH_HOST/FMH_PORT env (set above); an
# operator overrides either by re-setting the env or passing the flags.
CMD ["fusion-model-hub", "serve"]
