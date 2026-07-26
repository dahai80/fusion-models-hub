import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    @abstractmethod
    def model_version_dir(self, model_id: str, version: str) -> Path:
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
