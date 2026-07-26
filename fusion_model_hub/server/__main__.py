import argparse
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Fusion Model Hub — Model repository & management server")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"], help="Command to run")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--data-dir", default="", help="Data directory (default: ./data)")
    parser.add_argument("--db-url", default="", help="Database URL (default: sqlite in data-dir)")
    parser.add_argument("--mlx-url", default="http://localhost:11434", help="Fusion-MLX API URL")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    from .config import Settings
    settings = Settings(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        db_url=args.db_url,
        mlx_url=args.mlx_url,
        log_level=args.log_level,
    )

    uvicorn.run(
        "fusion_model_hub.server.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
