import os
from collections.abc import AsyncGenerator
from functools import cache
from typing import Literal

from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


@cache
def get_db_connection_string(connection_type: Literal['async', 'sync'] = 'async') -> str:
    db_connection_string = os.getenv("SUPABASE_DB_URL")

    if not db_connection_string:
        raise ValueError("The value for env variable SUPABASE_DB_URL is not set.")

    if db_connection_string.startswith("postgresql://") and connection_type == 'async':
        db_connection_string = db_connection_string.replace(
            "postgresql://", "postgresql+asyncpg://", 1)
    elif db_connection_string.startswith("postgresql://") and connection_type == "sync":
        return db_connection_string
    elif not db_connection_string.startswith("postgresql://"):
      raise ValueError(f"Invalid Database URl: {db_connection_string[:20]}")
    return db_connection_string


async def init_database() -> None:
    """Initialize database engine and session factory. Safe to call multiple times."""
    global _engine, _session_factory

    if _engine is not None:
        return

    _engine = create_async_engine(
        get_db_connection_string(),
        pool_size=int(os.environ.get("DB_POOL_SIZE", "5")),
        max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "10")),
        pool_pre_ping=True,  # Verify connections before use
    )
    _session_factory = async_sessionmaker(
        _engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )



async def close_database() -> None:
    """Dispose database engine and reset state. Safe to call multiple times."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_database_session()-> AsyncGenerator[AsyncSession, None]:

  if not _session_factory:
    raise RuntimeError("Database not initialized. Call init_database first.")

  async with _session_factory() as session:
    try:
      yield session
    except Exception:
      await session.rollback()
      raise
    finally:
        await session.close()


def initialize_app_checkpointer() -> None:
    """Create LangGraph checkpoint tables."""

    postgres_connection_string: str = get_db_connection_string("sync")

    if postgres_connection_string:
        with PostgresSaver.from_conn_string(conn_string=postgres_connection_string) as checkpointer:
            checkpointer.setup()


# async def create():
#     return await get_supabase_client()


# @staticmethod
# async def get_supabase_client() -> AsyncClient:
#     supabase_url = os.environ.get("SUPABASE_URL")
#     supabase_key = os.environ.get("SUPABASE_KEY")
#     if not supabase_url:
#         raise ValueError("SUPABASE_URL must be set")
#     if not supabase_key:
#         raise ValueError("SUPABASE_URL  SUPABASE_KEY must be set")
#     try:
#         return await create_async_client(supabase_url, supabase_key)
#     except AsyncSupabaseException as e:
#         raise AsyncSupabaseException("Exception raised by Supabase client.") from e
