"""FastAPI routes for authentication, agent control, and results endpoints.

Implements the web API layer for the blockchain stablecoin explorer.
"""

import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

from api.auth_middleware import (
    get_client_info,
    requires_permission,
)
from core.auth0_manager import (
    Auth0Error,
    Auth0Manager,
    UserInfo,
    get_auth0_manager,
)
from core.database import get_database
from core.db_manager import DatabaseManager, InvalidUUIDError

logger = logging.getLogger(__name__)

# ==================== Routers ====================

auth_router = APIRouter(tags=["Authentication"])
agent_router = APIRouter(prefix="/api/agent", tags=["Agent Control"])
results_router = APIRouter(prefix="/api/results", tags=["Results"])


# ==================== Request/Response Models ====================


class RunConfigRequest(BaseModel):
    """Request model for triggering an agent run."""

    max_records_per_explorer: Optional[int] = Field(
        default=None,
        ge=1,
        le=100000,
        description="Maximum records to collect per explorer",
    )
    explorers: Optional[List[str]] = Field(
        default=None,
        description="List of explorer names to query (default: all configured)",
    )
    stablecoins: Optional[List[str]] = Field(
        default=None,
        description="List of stablecoins to collect (default: all configured)",
    )


class RunResponse(BaseModel):
    """Response model for agent run creation."""

    run_id: str
    status: str
    message: str
    started_at: str


class StatusResponse(BaseModel):
    """Response model for run status."""

    run_id: str
    status: str
    progress: Optional[float] = None
    progress_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class ResultSummary(BaseModel):
    """Summary model for a run result."""

    run_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_records: Optional[int] = None
    explorers_queried: Optional[List[str]] = None


class ResultDetails(BaseModel):
    """Detailed result model."""

    run_id: str
    user_id: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress: Optional[float] = None
    progress_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class TokenResponse(BaseModel):
    """Response model for token exchange."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    user_id: Optional[str] = None
    email: Optional[str] = None


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


# ==================== Session State Storage ====================

# In-memory state storage for OAuth flow (use Redis in production)
_oauth_states: Dict[str, datetime] = {}


def _generate_state() -> str:
    """Generate a secure random state for OAuth."""
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = datetime.now(timezone.utc)
    return state


def _validate_state(state: str) -> bool:
    """Validate and consume an OAuth state."""
    if state not in _oauth_states:
        return False
    
    created_at = _oauth_states.pop(state)
    # State expires after 10 minutes
    age = (datetime.now(timezone.utc) - created_at).total_seconds()
    return age < 600


def _cleanup_expired_states() -> None:
    """Remove expired OAuth states."""
    now = datetime.now(timezone.utc)
    expired = [
        state for state, created_at in _oauth_states.items()
        if (now - created_at).total_seconds() > 600
    ]
    for state in expired:
        _oauth_states.pop(state, None)


# ==================== Dependencies ====================


def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    db = get_database()
    return DatabaseManager(db)


# ==================== Authentication Endpoints ====================


@auth_router.get("/login", response_class=RedirectResponse)
async def login(
    auth0_manager: Auth0Manager = Depends(get_auth0_manager),
) -> RedirectResponse:
    """Redirect to Auth0 login page.
    
    Initiates the OAuth 2.0 authorization code flow by redirecting
    the user to Auth0's authorization endpoint.
    """
    _cleanup_expired_states()
    state = _generate_state()
    
    authorization_url = auth0_manager.get_authorization_url(state=state)
    logger.info("Redirecting user to Auth0 for authentication")
    
    return RedirectResponse(url=authorization_url, status_code=status.HTTP_302_FOUND)


@auth_router.get("/callback", response_model=TokenResponse)
async def callback(
    request: Request,
    code: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    error: Optional[str] = Query(default=None),
    error_description: Optional[str] = Query(default=None),
    auth0_manager: Auth0Manager = Depends(get_auth0_manager),
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> TokenResponse:
    """Handle Auth0 callback after authentication.
    
    Exchanges the authorization code for tokens and creates/updates
    the user in the database.
    """
    # Handle OAuth errors
    if error:
        logger.warning(f"OAuth error: {error} - {error_description}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authentication failed: {error_description or error}",
        )
    
    # Validate required parameters
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code",
        )
    
    # Validate state parameter (CSRF protection)
    if state and not _validate_state(state):
        logger.warning("Invalid or expired OAuth state")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state parameter",
        )
    
    try:
        # Exchange code for tokens
        tokens = await auth0_manager.exchange_code_for_tokens(code)
        
        access_token = tokens.get("access_token")
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No access token received from Auth0",
            )
        
        # Verify the token and get user info
        user_info = await auth0_manager.verify_token(access_token)
        
        # Create or update user in database
        await db_manager.get_or_create_user(
            user_id=user_info.user_id,
            email=user_info.email or "",
            name=user_info.name,
        )
        
        # Log the login action
        client_info = get_client_info(request)
        await db_manager.log_user_action(
            user_id=user_info.user_id,
            action="login",
            ip_address=client_info.get("ip_address"),
            user_agent=client_info.get("user_agent"),
        )
        
        logger.info(f"User {user_info.user_id} authenticated successfully")
        
        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=tokens.get("expires_in"),
            user_id=user_info.user_id,
            email=user_info.email,
        )
        
    except Auth0Error as e:
        logger.error(f"Auth0 error during callback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication failed: {str(e)}",
        )


@auth_router.get("/logout", response_class=RedirectResponse)
async def logout(
    request: Request,
    return_to: Optional[str] = Query(default=None),
    auth0_manager: Auth0Manager = Depends(get_auth0_manager),
    user: Optional[UserInfo] = None,
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> RedirectResponse:
    """Log out the user and redirect to Auth0 logout.
    
    Clears the session and redirects to Auth0's logout endpoint
    to clear the Auth0 session as well.
    """
    logout_url = auth0_manager.get_logout_url(return_to=return_to)
    logger.info("User logged out, redirecting to Auth0 logout")
    
    return RedirectResponse(url=logout_url, status_code=status.HTTP_302_FOUND)


# ==================== Agent Control Endpoints ====================


async def run_agent_task(
    run_id: str,
    user_id: str,
    config: Optional[Dict[str, Any]],
    db_manager: DatabaseManager,
) -> None:
    """Background task to run the agent.
    
    This is a placeholder that will be replaced with actual agent
    orchestration when the collector components are implemented.
    """
    try:
        # Update status to running
        await db_manager.update_run_status(run_id, "running")
        await db_manager.update_run_progress(run_id, 0.0, "Starting data collection...")
        
        # TODO: Implement actual agent orchestration
        # For now, this is a placeholder that simulates progress
        logger.info(f"Agent run {run_id} started for user {user_id}")
        
        # The actual implementation will:
        # 1. Initialize collectors for each explorer
        # 2. Fetch stablecoin transactions in parallel
        # 3. Classify activities
        # 4. Aggregate and deduplicate data
        # 5. Export to JSON and save to database
        
        # Placeholder: Mark as completed with no results
        # This will be replaced when collectors are implemented
        await db_manager.update_run_progress(run_id, 1.0, "Collection complete")
        await db_manager.update_run_status(run_id, "completed")
        
        logger.info(f"Agent run {run_id} completed")
        
    except Exception as e:
        logger.error(f"Agent run {run_id} failed: {e}")
        await db_manager.update_run_status(
            run_id, "failed", error_message=str(e)
        )


@agent_router.post("/run", response_model=RunResponse)
async def trigger_agent_run(
    request: Request,
    background_tasks: BackgroundTasks,
    config: Optional[RunConfigRequest] = None,
    user: UserInfo = Depends(requires_permission("run:agent")),
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> RunResponse:
    """Trigger a new data collection run.
    
    Creates a new agent run and starts the data collection process
    in the background. Returns immediately with the run ID.
    
    Requires the 'run:agent' permission.
    """
    # Prepare run configuration
    run_config = config.model_dump(exclude_none=True) if config else None
    
    # Create the run in database
    run_id = await db_manager.create_run(
        user_id=user.user_id,
        config=run_config,
    )
    
    # Log the action
    client_info = get_client_info(request)
    await db_manager.log_user_action(
        user_id=user.user_id,
        action="run_agent",
        resource_type="agent_run",
        resource_id=run_id,
        ip_address=client_info.get("ip_address"),
        user_agent=client_info.get("user_agent"),
        details={"config": run_config},
    )
    
    # Start the agent in background
    background_tasks.add_task(
        run_agent_task,
        run_id=run_id,
        user_id=user.user_id,
        config=run_config,
        db_manager=db_manager,
    )
    
    started_at = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Agent run {run_id} triggered by user {user.user_id}")
    
    return RunResponse(
        run_id=run_id,
        status="pending",
        message="Agent run started successfully",
        started_at=started_at,
    )


@agent_router.get("/status/{run_id}", response_model=StatusResponse)
async def get_run_status(
    run_id: str,
    user: UserInfo = Depends(requires_permission("view:results")),
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> StatusResponse:
    """Get the status of an agent run.
    
    Returns the current status, progress, and any error messages
    for the specified run.
    
    Requires the 'view:results' permission.
    """
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
    
    # Check if user owns this run or has admin permissions
    if run.user_id != user.user_id and not user.has_permission("admin:config"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this run",
        )
    
    return StatusResponse(
        run_id=str(run.run_id),
        status=run.status.value,
        progress=run.progress,
        progress_message=run.progress_message,
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        error_message=run.error_message,
    )


# ==================== Results Endpoints ====================


@results_router.get("", response_model=List[ResultSummary])
async def list_results(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: UserInfo = Depends(requires_permission("view:results")),
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> List[ResultSummary]:
    """List all runs for the authenticated user.
    
    Returns a paginated list of run summaries, ordered by start time
    (most recent first).
    
    Requires the 'view:results' permission.
    """
    runs = await db_manager.get_user_runs(
        user_id=user.user_id,
        limit=limit,
        offset=offset,
    )
    
    results = []
    for run in runs:
        # Get result info if available
        result = await db_manager.get_run_result(str(run.run_id))
        
        results.append(ResultSummary(
            run_id=str(run.run_id),
            status=run.status.value,
            started_at=run.started_at.isoformat() if run.started_at else None,
            completed_at=run.completed_at.isoformat() if run.completed_at else None,
            total_records=result.total_records if result else None,
            explorers_queried=result.explorers_queried if result else None,
        ))
    
    return results


@results_router.get("/{run_id}", response_model=ResultDetails)
async def get_result_details(
    request: Request,
    run_id: str,
    user: UserInfo = Depends(requires_permission("view:results")),
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> ResultDetails:
    """Get detailed results for a specific run.
    
    Returns full run details including configuration, status,
    and result summary.
    
    Requires the 'view:results' permission.
    """
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
    
    # Check if user owns this run or has admin permissions
    if details["user_id"] != user.user_id and not user.has_permission("admin:config"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this run",
        )
    
    # Log the view action
    client_info = get_client_info(request)
    await db_manager.log_user_action(
        user_id=user.user_id,
        action="view_result",
        resource_type="agent_run",
        resource_id=run_id,
        ip_address=client_info.get("ip_address"),
        user_agent=client_info.get("user_agent"),
    )
    
    return ResultDetails(**details)


@results_router.get("/{run_id}/download")
async def download_result(
    request: Request,
    run_id: str,
    user: UserInfo = Depends(requires_permission("download:data")),
    db_manager: DatabaseManager = Depends(get_db_manager),
) -> FileResponse:
    """Download the JSON output file for a run.
    
    Returns the JSON file containing all collected data for the
    specified run.
    
    Requires the 'download:data' permission.
    """
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
    
    # Check if user owns this run or has admin permissions
    if details["user_id"] != user.user_id and not user.has_permission("admin:config"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to download this run",
        )
    
    # Check if run has completed with results
    if details["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run is not completed (status: {details['status']})",
        )
    
    result = details.get("result")
    if not result or not result.get("output_file_path"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No output file available for this run",
        )
    
    file_path = Path(result["output_file_path"])
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file not found on server",
        )
    
    # Log the download action
    client_info = get_client_info(request)
    await db_manager.log_user_action(
        user_id=user.user_id,
        action="download_result",
        resource_type="agent_run",
        resource_id=run_id,
        ip_address=client_info.get("ip_address"),
        user_agent=client_info.get("user_agent"),
    )
    
    logger.info(f"User {user.user_id} downloading result for run {run_id}")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/json",
    )
