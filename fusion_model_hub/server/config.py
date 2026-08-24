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
    download_speed_limit_kbps: int = 0
    precision_loss_threshold: float = 10.0
    mlx_internal_api_key: str = ""
    cache_dir: str = ""
    api_key_pepper: str = ""
    expose_metrics: bool = False

    def __post_init__(self):
        if not self.cache_dir:
            self.cache_dir = os.environ.get("FMH_CACHE_DIR", str(os.path.join(self.data_dir or os.getcwd(), "cache")))
        if not self.data_dir:
            self.data_dir = os.environ.get("FMH_DATA_DIR", str(os.path.join(os.getcwd(), "data")))
        if not self.db_url:
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
            self.cors_origins = [
                o.strip() for o in cors_env.split(",") if o.strip()
            ]
        if not self.mlx_internal_api_key:
            import logging
            logger = logging.getLogger(__name__)
            if os.environ.get("FUSION_MLX_API_KEY"):
                self.mlx_internal_api_key = os.environ["FUSION_MLX_API_KEY"]
                logger.info("MLX API key loaded from env FUSION_MLX_API_KEY")
            elif os.environ.get("MLX_INTERNAL_API_KEY"):
                self.mlx_internal_api_key = os.environ["MLX_INTERNAL_API_KEY"]
                logger.warning(
                    "MLX API key loaded from deprecated env MLX_INTERNAL_API_KEY — "
                    "migrate to FUSION_MLX_API_KEY"
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
                                mlx_settings, mode,
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
                "FUSION_MLX_API_KEY not set — "
                "Hub→MLX requests will have no Bearer token"
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
