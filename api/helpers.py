"""Helper functions for API routes to maintain DRY principles."""

from typing import Any, Dict

from fastapi import HTTPException, status

from core.auth0_manager import UserInfo
from core.db_manager import DatabaseManager, InvalidUUIDError
from models.database import AgentRun


async def get_authorized_run(
    run_id: str, user: UserInfo, db_manager: DatabaseManager, action: str = "view"
) -> AgentRun:
    """Fetch a run and verify the user is authorized to access it."""
    try:
        run = await db_manager.get_run(run_id)
    except InvalidUUIDError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid run ID format",
        )

    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    if run.user_id != user.user_id and not user.has_permission("admin:config"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have permission to {action} this run",
        )

    return run


async def get_authorized_run_details(
    run_id: str, user: UserInfo, db_manager: DatabaseManager, action: str = "view"
) -> Dict[str, Any]:
    """Fetch run details and verify the user is authorized to access it."""
    try:
        details = await db_manager.get_run_details(run_id)
    except InvalidUUIDError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid run ID format",
        )

    if details is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    user_id = details.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Run details missing user_id",
        )

    if user_id != user.user_id and not user.has_permission("admin:config"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You don't have permission to {action} this run",
        )

    return details
