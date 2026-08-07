from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_session_factory as _make_session_factory
from ..storage.base import StorageBackend
from ..storage.local_store import LocalStore
from .config import Settings

_settings: Settings | None = None
_session_factory = None
_store: StorageBackend | None = None
_cache = None


def init_deps(settings: Settings, engine) -> None:
    global _settings, _session_factory, _store, _cache
    _settings = settings
    _session_factory = _make_session_factory(engine)
    if settings.storage_type == "minio" and settings.minio_endpoint:
        from ..storage.minio_store import MinioStore
        _store = MinioStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )
    else:
        _store = LocalStore(data_dir=settings.data_dir)
    from ..cache.manager import CacheManager
    _cache = CacheManager(cache_root=settings.cache_dir)


def get_settings() -> Settings:
    if _settings is None:
        return Settings()
    return _settings


async def get_session() -> Generator[AsyncSession, None, None]:
    if _session_factory is None:
        raise RuntimeError("Dependencies not initialized — call init_deps() first")
    async with _session_factory() as session:
        yield session


def get_session_factory():
    if _session_factory is None:
        raise RuntimeError("Dependencies not initialized — call init_deps() first")
    return _session_factory


def get_store() -> StorageBackend:
    if _store is None:
        raise RuntimeError("Dependencies not initialized — call init_deps() first")
    return _store


def get_cache_manager():
    if _cache is None:
        raise RuntimeError("Dependencies not initialized — call init_deps() first")
    return _cache


SessionDep = Annotated[AsyncSession, Depends(get_session)]
StoreDep = Annotated[StorageBackend, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
CacheDep = Annotated[object, Depends(get_cache_manager)]
