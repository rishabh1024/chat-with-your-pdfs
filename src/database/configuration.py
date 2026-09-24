import logging
from collections.abc import AsyncGenerator
from functools import cache
from typing import Literal
from urllib.parse import urlsplit

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.settings import load_environment_variables

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

settings = load_environment_variables()
logger = logging.getLogger(__name__)


@cache
def get_db_connection_string(connection_type: Literal["async", "sync"] = "async") -> str:
    db_connection_string = settings.supabase.db_url

    if not db_connection_string:
        raise ValueError("The value for env variable SUPABASE_DB_URL is not set.")

    if db_connection_string.startswith("postgresql://") and connection_type == "async":
        db_connection_string = db_connection_string.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
    elif db_connection_string.startswith("postgresql://") and connection_type == "sync":
        return db_connection_string
    elif not db_connection_string.startswith("postgresql://"):
        scheme = urlsplit(db_connection_string).scheme or "missing"
        raise ValueError(f"Invalid database URL scheme: {scheme}")
    return db_connection_string


async def init_database() -> None:
    """Initialize database engine and session factory. Safe to call multiple times."""
    global _engine, _session_factory

    if _engine is not None:
        logger.debug("database.initialize.skipped status=already_initialized")
        return

    try:
        _engine = create_async_engine(
            get_db_connection_string(),
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,
        )
        _session_factory = async_sessionmaker(
            _engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )
    except Exception as error:
        _engine = None
        _session_factory = None
        logger.error(
            "database.initialize.failed error_type=%s",
            type(error).__name__,
        )
        raise

    logger.info("database.initialize.completed")


async def close_database() -> None:
    """Dispose database engine and reset state. Safe to call multiple times."""
    global _engine, _session_factory

    if _engine is not None:
        try:
            await _engine.dispose()
        except Exception as error:
            logger.error(
                "database.shutdown.failed error_type=%s",
                type(error).__name__,
            )
            raise
        else:
            _engine = None
            _session_factory = None
            logger.info("database.shutdown.completed")
    else:
        logger.debug("database.shutdown.skipped status=not_initialized")


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:

    if not _session_factory:
        raise RuntimeError("Database not initialized. Call init_database first.")

    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            logger.debug("database.session.rollback")
            await session.rollback()
            raise
        finally:
            await session.close()
