from typing import Annotated, Generator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_session_factory as _make_session_factory
from ..storage.local_store import LocalStore
from .config import Settings

_settings: Settings | None = None
_session_factory = None
_store: LocalStore | None = None


def init_deps(settings: Settings, engine) -> None:
    global _settings, _session_factory, _store
    _settings = settings
    _session_factory = _make_session_factory(engine)
    _store = LocalStore(data_dir=settings.data_dir)


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


def get_store() -> LocalStore:
    if _store is None:
        raise RuntimeError("Dependencies not initialized — call init_deps() first")
    return _store


SessionDep = Annotated[AsyncSession, Depends(get_session)]
StoreDep = Annotated[LocalStore, Depends(get_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
