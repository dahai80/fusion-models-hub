import logging
import threading
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base

logger = logging.getLogger(__name__)

# Track every async engine created in this process so they can be disposed
# deterministically. aiosqlite spawns a background worker thread bound to the
# event loop; if the engine is never disposed, the worker wakes after the loop
# closes and raises `RuntimeError: Event loop is closed`. This registry lets
# tests (conftest) and the server lifespan dispose all engines before teardown.
_engines: list = []
# R-P2/#10: guard the engine registry. aiosqlite worker threads + concurrent
# get_engine calls (tests spinning many sessions in parallel) raced on append;
# dispose_all_engines reassigned the list mid-append, losing an engine that then
# never got disposed — `RuntimeError: Event loop is closed` at teardown. One
# process-wide lock makes append + drain atomic.
_engines_lock = threading.Lock()

# R9: SQLite default journal mode is DELETE — readers block the single writer
# and a contended write fails immediately with "database is locked". WAL allows
# concurrent readers alongside one writer; busy_timeout makes a contended write
# wait up to N ms instead of erroring. Applied only to file-backed SQLite
# (in-memory DBs ignore these PRAGMAs and cannot use WAL).
_SQLITE_PRAGMAS = [
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("busy_timeout", "5000"),
    ("foreign_keys", "ON"),
]


def _is_file_sqlite(db_url: str) -> bool:
    return db_url.startswith("sqlite") and ":memory:" not in db_url


def _is_server_db(db_url: str) -> bool:
    # P1-15: only PostgreSQL/MySQL/others use a real connection pool (QueuePool).
    # aiosqlite uses NullPool (one connection); passing pool_size there errors.
    return not db_url.startswith("sqlite")


def get_engine(db_url: str = "", *, pool_size: int = 10, max_overflow: int = 20):
    if not db_url:
        db_url = "sqlite+aiosqlite:///./data/hub.db"
    engine_kwargs: dict = {"echo": False, "pool_pre_ping": True}
    # P1-15: expose pool sizing for server-side DBs. SQLite ignores these
    # (NullPool); only QueuePool-backed dialects honor pool_size/max_overflow.
    if _is_server_db(db_url):
        engine_kwargs["pool_size"] = pool_size
        engine_kwargs["max_overflow"] = max_overflow
    engine = create_async_engine(db_url, **engine_kwargs)
    with _engines_lock:
        _engines.append(engine)
    if _is_file_sqlite(db_url):
        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _record):
            try:
                cur = dbapi_conn.cursor()
                for name, value in _SQLITE_PRAGMAS:
                    cur.execute(f"PRAGMA {name}={value}")
                cur.close()
            except Exception as e:
                logger.warning("Failed to set SQLite PRAGMA: %s", e)
        logger.info("SQLite WAL+busy_timeout enabled for %s", db_url)
    return engine


def get_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db(engine) -> None:
    db_path = engine.url.database
    if db_path and db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized: %s", engine.url)


async def dispose_all_engines() -> None:
    global _engines
    with _engines_lock:
        pending = _engines
        _engines = []
    for engine in pending:
        try:
            await engine.dispose()
        except Exception as e:
            logger.warning("Failed to dispose engine %s: %s", engine.url, e)
    logger.info("Disposed %d async engine(s)", len(pending))
