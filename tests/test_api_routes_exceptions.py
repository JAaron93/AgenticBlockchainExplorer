import pytest
import logging
from contextlib import ExitStack
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from fastapi import HTTPException
from api.routes import download_result

@pytest.mark.asyncio
async def test_download_result_fallback_on_config_error():
    # Setup mocks
    mock_request = MagicMock()
    mock_user = MagicMock()
    mock_user.user_id = "user123"
    mock_db = AsyncMock()
    
    # Mock get_authorized_run_details
    mock_details = {
        "status": "completed",
        "result": {
            "output_file_path": "/tmp/output.json"
        }
    }
    
    with ExitStack() as stack:
        # Patch all components to avoid nested 'with' blocks
        stack.enter_context(patch("api.routes.get_authorized_run_details", return_value=mock_details))
        mock_logger = stack.enter_context(patch("api.routes.logger"))
        mock_get_config = stack.enter_context(patch("main.get_config"))
        MockHandler = stack.enter_context(patch("core.security.safe_path_handler.SafePathHandler"))
        stack.enter_context(patch("pathlib.Path.exists", return_value=True))
        MockResponse = stack.enter_context(patch("api.routes.FileResponse"))
        
        # Setup specific mock behaviors
        mock_get_config.side_effect = AttributeError("output")
        mock_handler = MockHandler.return_value
        mock_handler.validate_path.return_value = True
        mock_handler.base_directory = Path("./output").resolve()
        
        # Execute test
        await download_result(
            request=mock_request,
            run_id="run123",
            user=mock_user,
            db_manager=mock_db
        )
        
        # Check that warning was logged
        mock_logger.warning.assert_called_with(
            "Configuration access failed for download_result; falling back to Path('./output').resolve()"
        )
        # Check that SafePathHandler was initialized with resolve of ./output
        MockHandler.assert_called_with(Path("./output").resolve())

        # Verify FileResponse was called with correctly resolved path
        expected_path = Path(mock_details["result"]["output_file_path"]).resolve()
        MockResponse.assert_called_once_with(
            path=expected_path,
            filename=expected_path.name,
            media_type="application/json"
        )

if __name__ == "__main__":
    pytest.main([__file__])
