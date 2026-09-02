import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 11444
    data_dir: str = ""
    db_url: str = ""
    mlx_url: str = "http://127.0.0.1:11434"
    log_level: str = "INFO"
    # E-E11: default CORS to an empty origin list (deny all cross-origin).
    # The prior ["*"] let any web origin read API responses. Same-origin and
    # non-browser clients (curl/fusion-cli/SDK) are unaffected by CORS; only
    # browser XHR/fetch from another origin is gated. An operator opts in to
    # specific origins via FMH_CORS_ORIGINS (comma-separated) in __post_init__.
    cors_origins: list[str] = field(default_factory=list)
    max_upload_size_mb: int = 50000  # 50GB
    auth_enabled: bool = True
    storage_type: str = "local"
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "fusion-models"
    minio_secure: bool = True
    backup_dir: str = ""
    backup_interval_seconds: int = 86400
    tls_certfile: str = ""
    tls_keyfile: str = ""
    bench_url: str = ""
    bench_auto_trigger: bool = False
    # #3: Bearer/X-API-Key for Hub→Fusion-Bench requests. Fusion-Bench gates
    # task creation behind Permission.TASK_CREATE (anonymous VIEWER lacks it),
    # so the Hub must present an operator/admin bench API key to submit evals
    # or trigger benchmarks. Empty = attempt anonymous (will 403 if bench
    # enforces auth, surfaced as a clear FAILED on the eval row).
    bench_api_key: str = ""
    # #3: gate the async evaluation runner. When False, POST /evaluations
    # creates the PENDING row but does not spawn the runner (the row stays
    # PENDING until an operator or external bench callback updates it). Tests
    # set this False to avoid background tasks racing the in-memory DB; prod
    # leaves it True (default).
    eval_runner_enabled: bool = True
    download_speed_limit_kbps: int = 0
    precision_loss_threshold: float = 10.0
    mlx_internal_api_key: str = ""
    cache_dir: str = ""
    api_key_pepper: str = ""
    expose_metrics: bool = False
    # E-E6: optional shared secret that the very first (bootstrap) API key
    # creation must present. POST /auth/keys is public while zero active keys
    # exist, so without this anyone who can reach the Hub can race to create
    # the root admin key. If unset, bootstrap is still open but IP rate-limited
    # (see routers/auth.py) — set the env in any shared/networked deployment.
    auth_bootstrap_token: str = ""
    # P1-15: pool sizing for server-side DBs (PostgreSQL/MySQL). SQLite uses
    # NullPool (single connection) and ignores these. Before, the engine was a
    # fixed pool of 5+10 with pre_ping, undocumented and unconfigurable; a
    # multi-worker/fronted-by-gateway deployment would exhaust it. Expose the
    # knobs so an operator can match pool size to expected concurrency.
    db_pool_size: int = 10
    db_max_overflow: int = 20
    # #53: enforce gateway-origin + authoritative tenant on inbound requests.
    # When True, a request to /api/v1/* MUST carry X-Fusion-Route: gateway-decision
    # (the gateway stamps this on every outbound request) so direct-port access
    # cannot bypass the gateway's tenant derivation. X-Fusion-Tenant, when
    # present, overrides the api_key's tenant_id as the authoritative tenant for
    # the request (the gateway derives it from the key->team binding). Default
    # OFF so single-tenant dev / direct-CLI access keeps working; an operator
    # enables it in any gateway-fronted, multi-tenant deployment.
    gateway_origin_enforced: bool = False

    def __post_init__(self):
        # P1-21: wire FMH_HOST/FMH_PORT/FMH_LOG_LEVEL/FMH_DB_URL. Previously the
        # CLI serve parser passed non-empty defaults (127.0.0.1, 11444, INFO)
        # straight into Settings(), so __post_init__'s `if not self.x` env hooks
        # never fired — an operator who set FMH_DB_URL in the container env got
        # the derived SQLite path anyway. Resolve env here for the fields that
        # had no hook at all. Constructor-arg values (non-empty) still win.
        if not self.host or self.host == "127.0.0.1":
            env_host = os.environ.get("FMH_HOST", "")
            if env_host:
                self.host = env_host
        if not self.port or self.port == 11444:
            env_port = os.environ.get("FMH_PORT", "")
            if env_port:
                try:
                    self.port = int(env_port)
                except ValueError:
                    import logging

                    logging.getLogger(__name__).warning(
                        "FMH_PORT=%r is not an int; keeping default 11444",
                        env_port,
                    )
        if not self.log_level or self.log_level == "INFO":
            env_log = os.environ.get("FMH_LOG_LEVEL", "")
            if env_log:
                self.log_level = env_log
        if not self.cache_dir:
            self.cache_dir = os.environ.get("FMH_CACHE_DIR", str(os.path.join(self.data_dir or os.getcwd(), "cache")))
        if not self.data_dir:
            self.data_dir = os.environ.get("FMH_DATA_DIR", str(os.path.join(os.getcwd(), "data")))
        if not self.db_url:
            # P1-21: prefer an explicit FMH_DB_URL env (PostgreSQL/MySQL in a
            # container) before deriving the SQLite path. Before this, the env
            # was silently ignored — a container with FMH_DB_URL=postgresql://...
            # still wrote to ./data/hub.db.
            env_db = os.environ.get("FMH_DB_URL", "")
            if env_db:
                self.db_url = env_db
            else:
                self.db_url = f"sqlite+aiosqlite:///{os.path.join(self.data_dir, 'hub.db')}"
        if not self.mlx_url:
            self.mlx_url = os.environ.get("FMH_MLX_URL", "http://127.0.0.1:11434")
        if not self.storage_type:
            self.storage_type = os.environ.get("FMH_STORAGE_TYPE", "local")
        if not self.minio_endpoint:
            self.minio_endpoint = os.environ.get("FMH_MINIO_ENDPOINT", "")
        if not self.minio_access_key:
            self.minio_access_key = os.environ.get("FMH_MINIO_ACCESS_KEY", "")
        if not self.minio_secret_key:
            self.minio_secret_key = os.environ.get("FMH_MINIO_SECRET_KEY", "")
        if not self.backup_dir:
            self.backup_dir = os.environ.get("FMH_BACKUP_DIR", "")
        if not self.tls_certfile:
            self.tls_certfile = os.environ.get("FMH_TLS_CERTFILE", "")
        if not self.tls_keyfile:
            self.tls_keyfile = os.environ.get("FMH_TLS_KEYFILE", "")
        if not self.bench_url:
            self.bench_url = os.environ.get("FMH_BENCH_URL", "http://localhost:8090")
        if not self.bench_auto_trigger:
            self.bench_auto_trigger = os.environ.get("FMH_BENCH_AUTO_TRIGGER", "false").lower() == "true"
        if not self.bench_api_key:
            self.bench_api_key = os.environ.get("FMH_BENCH_API_KEY", "")
        # #3: FMH_EVAL_RUNNER_ENABLED=false keeps POST /evaluations from spawning
        # the async Fusion-Bench runner (row stays PENDING). Tests pass the ctor
        # arg directly; operators toggle via env.
        if os.environ.get("FMH_EVAL_RUNNER_ENABLED", "").lower() in ("false", "0", "no"):
            self.eval_runner_enabled = False
        if not self.download_speed_limit_kbps:
            self.download_speed_limit_kbps = int(os.environ.get("FMH_DOWNLOAD_SPEED_LIMIT", "0"))
        if not self.precision_loss_threshold:
            self.precision_loss_threshold = float(os.environ.get("FMH_PRECISION_LOSS_THRESHOLD", "10.0"))
        auth_env = os.environ.get("MODEL_HUB_AUTH_ENABLED", "").lower()
        if auth_env in ("false", "0", "no"):
            self.auth_enabled = False
        elif auth_env in ("true", "1", "yes"):
            self.auth_enabled = True
        # E-E11: wire the documented FMH_CORS_ORIGINS env (comma-separated).
        # Previously the env was documented in README/CLAUDE.md but never read,
        # so the only way to change CORS was to construct Settings() in code —
        # impossible for the server entry point. Env takes precedence over the
        # empty default; a literal "*" re-enables the legacy permissive mode.
        cors_env = os.environ.get("FMH_CORS_ORIGINS", "")
        if cors_env:
            self.cors_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        if not self.mlx_internal_api_key:
            import logging

            logger = logging.getLogger(__name__)
            if os.environ.get("FUSION_MLX_API_KEY"):
                self.mlx_internal_api_key = os.environ["FUSION_MLX_API_KEY"]
                logger.info("MLX API key loaded from env FUSION_MLX_API_KEY")
            elif os.environ.get("MLX_INTERNAL_API_KEY"):
                self.mlx_internal_api_key = os.environ["MLX_INTERNAL_API_KEY"]
                logger.warning(
                    "MLX API key loaded from deprecated env MLX_INTERNAL_API_KEY — migrate to FUSION_MLX_API_KEY"
                )
        if not self.mlx_internal_api_key:
            # Env unset — fall back to the Fusion-MLX server's own settings so a
            # local install works without separately configuring the key. MLX
            # gates on ~/.fusion-mlx/settings.json auth.api_key; without a
            # matching Bearer every hub→MLX call 401s (mlx_metrics empty, model
            # never loads, cluster nodes inactive). Mirror the resolution order
            # fusion-mlx/start.sh uses.
            import json
            import logging

            logger = logging.getLogger(__name__)
            mlx_settings = os.path.expanduser("~/.fusion-mlx/settings.json")
            try:
                with open(mlx_settings) as f:
                    key = json.load(f).get("auth", {}).get("api_key", "")
                if key:
                    # E-S13: warn if the settings file is world/group-readable —
                    # it holds the MLX auth key. We do not refuse to load (the
                    # install would break) but surface the lax perms loudly.
                    try:
                        mode = os.stat(mlx_settings).st_mode & 0o777
                        if mode & 0o077:
                            logger.warning(
                                "MLX settings file %s is group/other accessible "
                                "(mode %o); tighten to 0600 to protect the auth key",
                                mlx_settings,
                                mode,
                            )
                    except OSError:
                        pass
                    self.mlx_internal_api_key = key
                    logger.info("MLX API key loaded from %s", mlx_settings)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning("Failed to read MLX settings %s: %s", mlx_settings, e)
        if not self.mlx_internal_api_key:
            import logging

            logging.getLogger(__name__).warning(
                "FUSION_MLX_API_KEY not set — Hub→MLX requests will have no Bearer token"
            )
        if not self.api_key_pepper:
            import hashlib
            import logging

            pepper_logger = logging.getLogger(__name__)
            env_pepper = os.environ.get("FMH_API_KEY_PEPPER", "")
            if env_pepper:
                self.api_key_pepper = env_pepper
                pepper_logger.info("API key pepper loaded from env FMH_API_KEY_PEPPER")
            else:
                # E-S4: derive a per-install pepper so API-key hashes are never
                # bare SHA-256 even on a fresh install. The DB alone is useless
                # for offline cracking without this secret. Rotating env
                # FMH_API_KEY_PEPPER invalidates all keys (rotate keys after).
                mac = hashlib.sha256(f"fmh-pepper|{self.data_dir}".encode()).digest()
                self.api_key_pepper = mac.hex()
                pepper_logger.warning(
                    "FMH_API_KEY_PEPPER not set — derived install-local pepper "
                    "(acceptable for single-node dev; set the env for production)"
                )
        # E-S11: /metrics leaks internal request/latency telemetry. Default to
        # OFF — an operator must explicitly opt in via FMH_EXPOSE_METRICS=true.
        # When off, the /metrics route 404s even with auth disabled.
        if os.environ.get("FMH_EXPOSE_METRICS", "").lower() in ("true", "1", "yes"):
            self.expose_metrics = True
        if not self.auth_bootstrap_token:
            self.auth_bootstrap_token = os.environ.get("FMH_AUTH_BOOTSTRAP_TOKEN", "")
        # P1-15: wire pool sizing env so operators can tune without code edits.
        if not self.db_pool_size:
            self.db_pool_size = int(os.environ.get("FMH_DB_POOL_SIZE", "10"))
        if not self.db_max_overflow:
            self.db_max_overflow = int(os.environ.get("FMH_DB_MAX_OVERFLOW", "20"))
        # #53: wire FMH_GATEWAY_ORIGIN_ENFORCED (default false for single-tenant
        # dev). Operators enable it in gateway-fronted multi-tenant deployments.
        if os.environ.get("FMH_GATEWAY_ORIGIN_ENFORCED", "").lower() in ("true", "1", "yes"):
            self.gateway_origin_enforced = True
