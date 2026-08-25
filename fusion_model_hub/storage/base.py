import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    @abstractmethod
    def model_version_dir(self, model_id: str, version: str) -> Path:
        ...

    # E-R3: the ABC previously omitted write_chunk/assemble_chunks/models_dir
    # even though LocalStore implements them and versions.py calls them. The
    # omission hid that MinioStore never implemented them, so
    # FMH_STORAGE_TYPE=minio hit AttributeError on the first chunked upload and
    # on export/import. Declaring them on the ABC makes the contract explicit
    # and forces each backend to either implement or raise NotImplementedError
    # in the open — no silent AttributeError 500.
    @abstractmethod
    async def write_chunk(self, upload_id: str, chunk_index: int, chunk_data: bytes) -> Path:
        ...

    @abstractmethod
    async def assemble_chunks(
        self, upload_id: str, target_dir: Path, filename: str, total_chunks: int,
    ) -> tuple[Path, str, int]:
        ...

    # E-R3: LocalStore exposes the models root as a Path attribute that
    # versions.py walks for tar export/import. Declaring it as an abstract
    # property makes the contract explicit; MinioStore raises
    # NotImplementedError because object storage has no walkable local dir.
    @property
    @abstractmethod
    def models_dir(self) -> Path:
        ...

    @abstractmethod
    async def write_file(self, target_dir: Path, filename: str, data: bytes) -> tuple[Path, str, int]:
        ...

    @abstractmethod
    def get_file(self, file_path: str) -> Path | None:
        ...

    @abstractmethod
    def delete_version_files(self, model_id: str, version: str) -> bool:
        ...

    @abstractmethod
    def delete_model_files(self, model_id: str) -> bool:
        ...

    @abstractmethod
    def get_storage_stats(self) -> dict[str, Any]:
        ...
