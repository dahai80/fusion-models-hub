import logging
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .models import Base

logger = logging.getLogger(__name__)

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


def get_engine(db_url: str = ""):
    if not db_url:
        db_url = "sqlite+aiosqlite:///./data/hub.db"
    engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
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
