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
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
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
