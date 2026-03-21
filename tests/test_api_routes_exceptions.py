import pytest
import logging
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
    
    with patch("api.routes.get_authorized_run_details", return_value=mock_details):
        # Mock main.get_config to raise AttributeError (simulating missing config.output.directory)
        with patch("api.routes.logger") as mock_logger:
            with patch("main.get_config") as mock_get_config:
                mock_get_config.side_effect = AttributeError("output")
                
                # Should fallback to Path("./output")
                # We need to mock SafePathHandler too because /tmp might not be in ./output
                with patch("core.security.safe_path_handler.SafePathHandler") as MockHandler:
                    mock_handler = MockHandler.return_value
                    mock_handler.validate_path.return_value = True
                    mock_handler.base_directory = Path("./output").resolve()
                    
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("api.routes.FileResponse") as MockResponse:
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

if __name__ == "__main__":
    pytest.main([__file__])
