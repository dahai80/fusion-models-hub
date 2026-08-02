import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 11444
    data_dir: str = ""
    db_url: str = ""
    mlx_url: str = "http://localhost:11434"
    log_level: str = "INFO"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 50000  # 50GB
    auth_enabled: bool = False
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

    def __post_init__(self):
        if not self.data_dir:
            self.data_dir = os.environ.get("FMH_DATA_DIR", str(os.path.join(os.getcwd(), "data")))
        if not self.db_url:
            self.db_url = f"sqlite+aiosqlite:///{os.path.join(self.data_dir, 'hub.db')}"
        if not self.mlx_url:
            self.mlx_url = os.environ.get("FMH_MLX_URL", "http://localhost:11434")
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
