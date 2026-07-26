import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8080
    data_dir: str = ""
    db_url: str = ""
    mlx_url: str = "http://localhost:11434"
    log_level: str = "INFO"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 50000  # 50GB
    auth_enabled: bool = False

    def __post_init__(self):
        if not self.data_dir:
            self.data_dir = os.environ.get("FMH_DATA_DIR", str(os.path.join(os.getcwd(), "data")))
        if not self.db_url:
            self.db_url = f"sqlite+aiosqlite:///{os.path.join(self.data_dir, 'hub.db')}"
        if not self.mlx_url:
            self.mlx_url = os.environ.get("FMH_MLX_URL", "http://localhost:11434")
