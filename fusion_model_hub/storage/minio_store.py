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

    # E-R3: object storage has no walkable local directory. versions.py uses
    # store.models_dir for tar export/import, which is fundamentally local-fs.
    # Raising NotImplementedError (surfaced as 501 at the call site) is honest;
    # silently returning a bogus Path would have versions.py write/read the
    # process CWD and corrupt file_path. Tar export/import over MinIO is a
    # separate feature, not a silent gap.
    @property
    def models_dir(self) -> Path:
        raise NotImplementedError(
            "MinioStore has no local models_dir — tar export/import is not "
            "supported with FMH_STORAGE_TYPE=minio; use the chunked/URL upload "
            "paths or export from a local-storage node."
        )

    async def write_chunk(self, upload_id: str, chunk_index: int, chunk_data: bytes) -> Path:
        # E-R3: store each chunk as an object under uploads/{upload_id}/ so
        # chunked upload works on MinIO the same way LocalStore writes .part
        # files. The assemble step reads them back in order, writes the final
        # object, and removes the chunk objects.
        from io import BytesIO

        client = self._get_client()
        object_name = f"uploads/{upload_id}/{chunk_index:06d}.part"
        client.put_object(
            self.bucket,
            object_name,
            BytesIO(chunk_data),
            len(chunk_data),
        )
        logger.info(
            "Wrote MinIO chunk: upload=%s index=%d size=%d",
            upload_id,
            chunk_index,
            len(chunk_data),
        )
        return Path(object_name)

    async def assemble_chunks(
        self,
        upload_id: str,
        target_dir: Path,
        filename: str,
        total_chunks: int,
    ) -> tuple[Path, str, int]:
        from io import BytesIO

        client = self._get_client()
        final_object = f"{target_dir}/{filename}"
        hasher = hashlib.sha256()
        total_size = 0
        # Assemble by reading each chunk object and appending to the final
        # object. MinIO has no client-side multipart compose that hashes, so we
        # stream chunk-by-chunk: write the first chunk as the final object, then
        # append subsequent chunks via put_object with the part-of-stream length.
        # put_object overwrites, so we accumulate into a BytesIO for correctness
        # on the typical model-file size (cap is enforced upstream by
        # max_upload_size_mb). For very large files this is memory-bound; a
        # true multipart compose is a follow-up, tracked separately.
        buf = BytesIO()
        try:
            for i in range(total_chunks):
                part_name = f"uploads/{upload_id}/{i:06d}.part"
                resp = client.get_object(self.bucket, part_name)
                try:
                    data = resp.read()
                finally:
                    resp.close()
                    resp.release_conn()
                buf.write(data)
                hasher.update(data)
                total_size += len(data)
            buf.seek(0)
            client.put_object(self.bucket, final_object, buf, total_size)
        finally:
            # Best-effort chunk cleanup; never leak .part objects on success.
            for i in range(total_chunks):
                try:
                    client.remove_object(self.bucket, f"uploads/{upload_id}/{i:06d}.part")
                except Exception:
                    logger.debug("Failed to remove chunk %d for upload %s", i, upload_id)
        file_hash = hasher.hexdigest()
        logger.info(
            "Assembled MinIO upload: id=%s object=%s size=%d hash=%s",
            upload_id,
            final_object,
            total_size,
            file_hash[:16],
        )
        return Path(final_object), file_hash, total_size

    async def write_file(self, target_dir: Path, filename: str, data: bytes) -> tuple[Path, str, int]:
        client = self._get_client()
        object_name = f"{target_dir}/{filename}"
        from io import BytesIO

        client.put_object(
            self.bucket,
            str(object_name),
            BytesIO(data),
            len(data),
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

    def put_lfs_object(self, oid: str, data: bytes) -> Path:
        raise NotImplementedError("Git LFS object upload not implemented for MinioStore")

    def get_lfs_object(self, oid: str) -> Path | None:
        raise NotImplementedError("Git LFS object download not implemented for MinioStore")

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

    def write_sidecar(self, model_id: str, version: str, filename: str, data: bytes) -> Path:
        # #1: a real MinIO implementation is a one-line put_object; leave it
        # NotImplementedError so the contract is explicit (object-storage
        # sidecar watermark is a separate feature, surfaced as 501 not 500).
        raise NotImplementedError(
            "Watermark sidecar write not implemented for MinioStore — "
            "use FMH_STORAGE_TYPE=local, or track the MinIO sidecar feature."
        )

    def read_sidecar(self, model_id: str, version: str, filename: str) -> bytes | None:
        raise NotImplementedError(
            "Watermark sidecar read not implemented for MinioStore — "
            "use FMH_STORAGE_TYPE=local, or track the MinIO sidecar feature."
        )

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
