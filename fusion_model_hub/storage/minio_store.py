import hashlib
import logging
from pathlib import Path
from typing import Any

from .base import StorageBackend

logger = logging.getLogger(__name__)


class MinioStore(StorageBackend):
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "fusion-models",
        secure: bool = True,
    ):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket = bucket
        self.secure = secure
        self._client = None
        logger.info("MinioStore configured: endpoint=%s bucket=%s secure=%s", endpoint, bucket, secure)

    def _get_client(self):
        if self._client is None:
            try:
                from minio import Minio
                self._client = Minio(
                    self.endpoint,
                    access_key=self.access_key,
                    secret_key=self.secret_key,
                    secure=self.secure,
                )
                if not self._client.bucket_exists(self.bucket):
                    self._client.make_bucket(self.bucket)
                    logger.info("Created bucket: %s", self.bucket)
            except ImportError:
                raise RuntimeError("minio package not installed. Run: pip install minio")
            except Exception:
                logger.exception("Failed to connect to MinIO: %s", self.endpoint)
                raise
        return self._client

    def model_version_dir(self, model_id: str, version: str) -> Path:
        return Path(f"{model_id}/{version}")

    async def write_file(self, target_dir: Path, filename: str, data: bytes) -> tuple[Path, str, int]:
        client = self._get_client()
        object_name = f"{target_dir}/{filename}"
        from io import BytesIO
        client.put_object(
            self.bucket, str(object_name),
            BytesIO(data), len(data),
        )
        file_hash = hashlib.sha256(data).hexdigest()
        logger.info("Wrote file to MinIO: %s size=%d", object_name, len(data))
        return Path(object_name), file_hash, len(data)

    def get_file(self, file_path: str) -> Path | None:
        client = self._get_client()
        try:
            client.stat_object(self.bucket, file_path)
            return Path(file_path)
        except Exception:
            return None

    def delete_version_files(self, model_id: str, version: str) -> bool:
        client = self._get_client()
        prefix = f"{model_id}/{version}/"
        deleted = 0
        for obj in client.list_objects(self.bucket, prefix=prefix, recursive=True):
            client.remove_object(self.bucket, obj.object_name)
            deleted += 1
        logger.info("Deleted %d objects from MinIO: prefix=%s", deleted, prefix)
        return deleted > 0

    def delete_model_files(self, model_id: str) -> bool:
        client = self._get_client()
        prefix = f"{model_id}/"
        deleted = 0
        for obj in client.list_objects(self.bucket, prefix=prefix, recursive=True):
            client.remove_object(self.bucket, obj.object_name)
            deleted += 1
        logger.info("Deleted %d objects from MinIO: prefix=%s", deleted, prefix)
        return deleted > 0

    def get_storage_stats(self) -> dict[str, Any]:
        client = self._get_client()
        total_size = 0
        file_count = 0
        model_count = 0
        seen_models = set()
        for obj in client.list_objects(self.bucket, recursive=True):
            file_count += 1
            total_size += obj.size or 0
            parts = obj.object_name.split("/")
            if parts:
                seen_models.add(parts[0])
        model_count = len(seen_models)
        return {
            "endpoint": self.endpoint,
            "bucket": self.bucket,
            "model_count": model_count,
            "file_count": file_count,
            "total_size_gb": round(total_size / (1024**3), 2),
        }
