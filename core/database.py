"""Database connection management with connection pooling.

Provides async database engine and session management for the application.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


from config.models import DatabaseConfig
from models.database import Base

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connections with connection pooling."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database connection manager.
        
        Args:
            config: Database configuration with connection URL and pool settings.
        """
        self._config = config
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine, creating it if necessary."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory, creating it if necessary."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._session_factory
    
    def _get_async_url(self, url: str) -> str:
        """Convert a synchronous database URL to async format.
        
        Args:
            url: Database URL (may be sync or async format).
            
        Returns:
            Async-compatible database URL.
        """
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url
    
    async def connect(self) -> None:
        """Initialize the database engine and session factory."""
        if self._engine is not None:
            logger.warning("Database already connected")
            return
        
        async_url = self._get_async_url(self._config.url)
        
        logger.info(f"Connecting to database with pool_size={self._config.pool_size}")
        
        self._engine = create_async_engine(
            async_url,
            pool_size=self._config.pool_size,
            max_overflow=self._config.max_overflow,
            pool_pre_ping=True,  # Enable connection health checks
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=False,          # Set to True for SQL debugging
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
        
        logger.info("Database connection established")
    
    async def disconnect(self) -> None:
        """Close the database engine and all connections."""
        if self._engine is not None:
            logger.info("Closing database connection")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection closed")
    
    async def create_tables(self) -> None:
        """Create all database tables.
        
        This should only be used for development/testing.
        Use Alembic migrations for production.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")
    
    async def drop_tables(self) -> None:
        """Drop all database tables.
        
        WARNING: This will delete all data. Use with caution.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session as a context manager.
        
        Yields:
            AsyncSession: Database session that will be automatically closed.
            
        Example:
            async with db.session() as session:
                result = await session.execute(query)
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def get_session(self) -> AsyncSession:
        """Get a new database session.
        
        Note: Caller is responsible for closing the session.
        Prefer using the session() context manager instead.
        
        Returns:
            AsyncSession: New database session.
        """
        return self.session_factory()


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None
_db_lock: Optional["asyncio.Lock"] = None

# Import asyncio for lock
import asyncio


def _get_lock() -> "asyncio.Lock":
    """Get or create the global database lock."""
    global _db_lock
    if _db_lock is None:
        _db_lock = asyncio.Lock()
    return _db_lock


def get_database() -> DatabaseConnection:
    """Get the global database connection instance.

    Returns:
        DatabaseConnection: The global database connection.

    Raises:
        RuntimeError: If database has not been initialized.
    """
    if _db_connection is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_connection


async def init_database(config: DatabaseConfig) -> DatabaseConnection:
    """Initialize the global database connection.

    This function is idempotent - if a connection already exists with the
    same configuration, it returns the existing connection. If a different
    configuration is provided, the old connection is gracefully closed
    before creating a new one.

    Thread-safe via asyncio.Lock to prevent races during concurrent calls.

    Args:
        config: Database configuration.

    Returns:
        DatabaseConnection: The initialized database connection.
    """
    global _db_connection

    async with _get_lock():
        # Return existing connection if it matches the config
        if _db_connection is not None:
            if _db_connection._config.url == config.url:
                logger.debug("Returning existing database connection")
                return _db_connection

            # Different config - close existing connection first
            logger.info("Closing existing connection for new configuration")
            await _db_connection.disconnect()
            _db_connection = None

        # Create new connection
        _db_connection = DatabaseConnection(config)
        return _db_connection


async def close_database() -> None:
    """Close the global database connection."""
    global _db_connection

    async with _get_lock():
        if _db_connection is not None:
            await _db_connection.disconnect()
            _db_connection = None
