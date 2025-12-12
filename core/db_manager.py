"""Database manager for CRUD operations.

Provides async methods for managing users, agent runs, results, and audit logs.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from sqlalchemy import select, update, desc

from models.database import User, AgentRun, RunResult, AuditLog, RunStatus
from core.database import DatabaseConnection

logger = logging.getLogger(__name__)


class InvalidUUIDError(ValueError):
    """Raised when an invalid UUID string is provided."""

    def __init__(self, param_name: str, value: str, original_error: Exception):
        self.param_name = param_name
        self.value = value
        self.original_error = original_error
        super().__init__(
            f"Invalid UUID for '{param_name}': '{value}' is not a valid UUID format"
        )


class InvalidStatusError(ValueError):
    """Raised when an invalid run status is provided."""

    def __init__(self, value: str, valid_statuses: list):
        self.value = value
        self.valid_statuses = valid_statuses
        super().__init__(
            f"Invalid status '{value}'. "
            f"Must be one of: {', '.join(valid_statuses)}"
        )


def _parse_uuid(value: str, param_name: str = "id") -> uuid.UUID:
    """Parse a string to UUID with proper error handling.

    Args:
        value: String value to parse as UUID.
        param_name: Name of the parameter for error messages.

    Returns:
        Parsed UUID object.

    Raises:
        InvalidUUIDError: If the value is not a valid UUID string.
    """
    if value is None:
        logger.error(f"UUID parsing failed: '{param_name}' is None")
        raise InvalidUUIDError(param_name, "None", TypeError("UUID cannot be None"))

    try:
        return uuid.UUID(value)
    except (ValueError, TypeError) as e:
        logger.error(
            f"UUID parsing failed for '{param_name}': "
            f"value='{value}', error={type(e).__name__}: {e}"
        )
        raise InvalidUUIDError(param_name, str(value), e) from e


class DatabaseManager:
    """Manager class for database CRUD operations."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize the database manager.

        Args:
            db_connection: Database connection instance.
        """
        self._db = db_connection

    # ==================== User Operations ====================

    async def get_or_create_user(
        self,
        user_id: str,
        email: str,
        name: Optional[str] = None
    ) -> User:
        """Get an existing user or create a new one.

        Args:
            user_id: Auth0 user ID.
            email: User's email address.
            name: User's display name.

        Returns:
            User instance.
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(User).where(User.user_id == user_id)
            )
            user = result.scalar_one_or_none()

            if user is None:
                user = User(
                    user_id=user_id,
                    email=email,
                    name=name,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(user)
                await session.flush()
                logger.info(f"Created new user: {user_id}")
            else:
                if user.email != email or user.name != name:
                    user.email = email
                    user.name = name
                    await session.flush()

            return user

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID.

        Args:
            user_id: Auth0 user ID.

        Returns:
            User instance or None if not found.
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(User).where(User.user_id == user_id)
            )
            return result.scalar_one_or_none()

    # ==================== Agent Run Operations ====================

    async def create_run(
        self,
        user_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new agent run.

        Args:
            user_id: ID of the user initiating the run.
            config: Optional run configuration.

        Returns:
            The run_id as a string.
        """
        run_id = uuid.uuid4()

        async with self._db.session() as session:
            run = AgentRun(
                run_id=run_id,
                user_id=user_id,
                status=RunStatus.PENDING,
                started_at=datetime.now(timezone.utc),
                config=config
            )
            session.add(run)
            await session.flush()
            logger.info(f"Created agent run: {run_id} for user: {user_id}")

        return str(run_id)

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of an agent run.

        Args:
            run_id: The run ID to update.
            status: New status ('pending', 'running', 'completed', 'failed').
            error_message: Optional error message for failed runs.

        Raises:
            InvalidStatusError: If status is not a valid RunStatus value.
            InvalidUUIDError: If run_id is not a valid UUID.
        """
        # Validate status before any DB operations
        valid_statuses = [s.value for s in RunStatus]
        try:
            validated_status = RunStatus(status)
        except ValueError:
            logger.error(
                f"Invalid status '{status}' for run {run_id}. "
                f"Valid statuses: {valid_statuses}"
            )
            raise InvalidStatusError(status, valid_statuses)

        parsed_run_id = _parse_uuid(run_id, "run_id")

        async with self._db.session() as session:
            values: Dict[str, Any] = {"status": validated_status}

            if status in ("completed", "failed"):
                values["completed_at"] = datetime.now(timezone.utc)

            if error_message:
                values["error_message"] = error_message

            await session.execute(
                update(AgentRun)
                .where(AgentRun.run_id == parsed_run_id)
                .values(**values)
            )
            logger.info(f"Updated run {run_id} status to: {status}")

    async def update_run_progress(
        self,
        run_id: str,
        progress: float,
        message: Optional[str] = None
    ) -> None:
        """Update the progress of an agent run.

        Args:
            run_id: The run ID to update.
            progress: Progress percentage (0.0 to 1.0).
            message: Optional progress message.
        """
        async with self._db.session() as session:
            values: Dict[str, Any] = {"progress": progress}
            if message:
                values["progress_message"] = message

            parsed_run_id = _parse_uuid(run_id, "run_id")
            await session.execute(
                update(AgentRun)
                .where(AgentRun.run_id == parsed_run_id)
                .values(**values)
            )

    async def get_run(self, run_id: str) -> Optional[AgentRun]:
        """Get an agent run by ID.

        Args:
            run_id: The run ID.

        Returns:
            AgentRun instance or None if not found.
        """
        parsed_run_id = _parse_uuid(run_id, "run_id")
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentRun).where(AgentRun.run_id == parsed_run_id)
            )
            return result.scalar_one_or_none()

    async def get_user_runs(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[AgentRun]:
        """Get all runs for a user.

        Args:
            user_id: The user ID.
            limit: Maximum number of runs to return.
            offset: Number of runs to skip.

        Returns:
            List of AgentRun instances.
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(AgentRun)
                .where(AgentRun.user_id == user_id)
                .order_by(desc(AgentRun.started_at))
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a run including results.

        Args:
            run_id: The run ID.

        Returns:
            Dictionary with run details and results, or None if not found.
        """
        parsed_run_id = _parse_uuid(run_id, "run_id")
        async with self._db.session() as session:
            run_result = await session.execute(
                select(AgentRun).where(AgentRun.run_id == parsed_run_id)
            )
            run = run_result.scalar_one_or_none()

            if run is None:
                return None

            result_query = await session.execute(
                select(RunResult).where(RunResult.run_id == parsed_run_id)
            )
            result = result_query.scalar_one_or_none()

            return {
                "run_id": str(run.run_id),
                "user_id": run.user_id,
                "status": run.status.value,
                "started_at": (
                    run.started_at.isoformat() if run.started_at else None
                ),
                "completed_at": (
                    run.completed_at.isoformat() if run.completed_at else None
                ),
                "config": run.config,
                "error_message": run.error_message,
                "progress": run.progress,
                "progress_message": run.progress_message,
                "result": {
                    "result_id": str(result.result_id),
                    "total_records": result.total_records,
                    "explorers_queried": result.explorers_queried,
                    "output_file_path": result.output_file_path,
                    "summary": result.summary,
                    "created_at": result.created_at.isoformat()
                } if result else None
            }

    # ==================== Run Result Operations ====================

    async def save_run_result(
        self,
        run_id: str,
        total_records: int,
        explorers_queried: List[str],
        output_file_path: str,
        summary: Dict[str, Any]
    ) -> str:
        """Save the results of an agent run.

        Args:
            run_id: The run ID.
            total_records: Total number of records collected.
            explorers_queried: List of explorer names queried.
            output_file_path: Path to the output JSON file.
            summary: Summary statistics dictionary.

        Returns:
            The result_id as a string.
        """
        result_id = uuid.uuid4()

        parsed_run_id = _parse_uuid(run_id, "run_id")
        async with self._db.session() as session:
            result = RunResult(
                result_id=result_id,
                run_id=parsed_run_id,
                total_records=total_records,
                explorers_queried=explorers_queried,
                output_file_path=output_file_path,
                summary=summary,
                created_at=datetime.now(timezone.utc)
            )
            session.add(result)
            await session.flush()
            logger.info(f"Saved result {result_id} for run {run_id}")

        return str(result_id)

    async def get_run_result(self, run_id: str) -> Optional[RunResult]:
        """Get the result for a run.

        Args:
            run_id: The run ID.

        Returns:
            RunResult instance or None if not found.
        """
        parsed_run_id = _parse_uuid(run_id, "run_id")
        async with self._db.session() as session:
            result = await session.execute(
                select(RunResult).where(RunResult.run_id == parsed_run_id)
            )
            return result.scalar_one_or_none()

    # ==================== Audit Log Operations ====================

    async def log_user_action(
        self,
        user_id: str,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a user action for audit purposes.

        Args:
            user_id: The user ID.
            action: Action performed (e.g., 'run_agent', 'download_result').
            resource_type: Type of resource affected (e.g., 'agent_run').
            resource_id: ID of the affected resource.
            ip_address: Client IP address.
            user_agent: Client user agent string.
            details: Additional details about the action.

        Returns:
            The log_id as a string.
        """
        log_id = uuid.uuid4()

        async with self._db.session() as session:
            log = AuditLog(
                log_id=log_id,
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                timestamp=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
                details=details
            )
            session.add(log)
            await session.flush()
            logger.debug(f"Logged action '{action}' for user {user_id}")

        return str(log_id)

    async def get_user_audit_logs(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLog]:
        """Get audit logs for a user.

        Args:
            user_id: The user ID.
            limit: Maximum number of logs to return.
            offset: Number of logs to skip.

        Returns:
            List of AuditLog instances.
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(AuditLog)
                .where(AuditLog.user_id == user_id)
                .order_by(desc(AuditLog.timestamp))
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def get_audit_logs_by_resource(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs for a specific resource.

        Args:
            resource_type: Type of resource.
            resource_id: ID of the resource.
            limit: Maximum number of logs to return.

        Returns:
            List of AuditLog instances.
        """
        async with self._db.session() as session:
            result = await session.execute(
                select(AuditLog)
                .where(
                    AuditLog.resource_type == resource_type,
                    AuditLog.resource_id == resource_id
                )
                .order_by(desc(AuditLog.timestamp))
                .limit(limit)
            )
            return list(result.scalars().all())
