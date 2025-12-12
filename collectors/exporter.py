"""JSON exporter for blockchain stablecoin data.

Exports aggregated data to JSON files and persists results to database.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from collectors.aggregator import AggregatedData

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from core.db_manager import DatabaseManager


# Use standard logging to avoid circular imports
logger = logging.getLogger(__name__)

# Agent version for metadata
AGENT_VERSION = "1.0.0"


class JSONExportError(Exception):
    """Raised when JSON export fails."""
    pass


class JSONSchemaValidationError(JSONExportError):
    """Raised when JSON schema validation fails."""
    pass


class JSONExporter:
    """Exports aggregated blockchain data to JSON files and database.

    Handles formatting, validation, and persistence of collected data.
    """

    # Required top-level keys in the output JSON
    REQUIRED_METADATA_KEYS = {
        "run_id",
        "collection_timestamp",
        "agent_version",
        "explorers_queried",
        "total_records",
    }

    REQUIRED_SUMMARY_KEYS = {
        "by_stablecoin",
        "by_activity_type",
        "by_chain",
    }

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        output_directory: str = "./output"
    ):
        """Initialize the JSON exporter.

        Args:
            db_manager: Database manager for persisting results.
            output_directory: Directory to write JSON files to.
        """
        self._db_manager = db_manager
        self._output_directory = output_directory

    def generate_filename(self, run_id: str) -> str:
        """Generate output filename with run_id and timestamp.

        Args:
            run_id: The unique run identifier.

        Returns:
            Filename in format: stablecoin_data_{run_id}_{timestamp}.json
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"stablecoin_data_{run_id}_{timestamp}.json"

    def _build_output_data(
        self,
        data: AggregatedData,
        run_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build the complete output data structure.

        Args:
            data: Aggregated data to export.
            run_id: The unique run identifier.
            user_id: Optional user ID who initiated the run.

        Returns:
            Complete output dictionary ready for JSON serialization.
        """
        collection_timestamp = datetime.now(timezone.utc).isoformat()

        metadata: Dict[str, Any] = {
            "run_id": run_id,
            "collection_timestamp": collection_timestamp,
            "agent_version": AGENT_VERSION,
            "explorers_queried": data.explorers_queried,
            "total_records": data.total_records,
        }

        if user_id:
            metadata["user_id"] = user_id

        output: Dict[str, Any] = {
            "metadata": metadata,
            "summary": {
                "by_stablecoin": {
                    coin: summary.to_dict()
                    for coin, summary in data.by_stablecoin.items()
                },
                "by_activity_type": data.by_activity_type,
                "by_chain": data.by_chain,
            },
            "transactions": [tx.to_dict() for tx in data.transactions],
            "holders": [h.to_dict() for h in data.holders],
        }

        if data.errors:
            output["errors"] = data.errors

        return output

    def validate_json_schema(self, data: Dict[str, Any]) -> bool:
        """Validate the JSON data structure before writing.

        Checks that all required fields are present and have valid types.

        Args:
            data: Dictionary to validate.

        Returns:
            True if validation passes.

        Raises:
            JSONSchemaValidationError: If validation fails.
        """
        # Check top-level structure
        if "metadata" not in data:
            raise JSONSchemaValidationError("Missing 'metadata' section")
        if "summary" not in data:
            raise JSONSchemaValidationError("Missing 'summary' section")
        if "transactions" not in data:
            raise JSONSchemaValidationError("Missing 'transactions' section")
        if "holders" not in data:
            raise JSONSchemaValidationError("Missing 'holders' section")

        # Validate metadata
        metadata = data["metadata"]
        missing_metadata = self.REQUIRED_METADATA_KEYS - set(metadata.keys())
        if missing_metadata:
            raise JSONSchemaValidationError(
                f"Missing metadata fields: {missing_metadata}"
            )

        # Validate metadata types
        if not isinstance(metadata["run_id"], str):
            raise JSONSchemaValidationError("'run_id' must be a string")
        if not isinstance(metadata["collection_timestamp"], str):
            raise JSONSchemaValidationError(
                "'collection_timestamp' must be a string"
            )
        if not isinstance(metadata["explorers_queried"], list):
            raise JSONSchemaValidationError(
                "'explorers_queried' must be a list"
            )
        if not isinstance(metadata["total_records"], int):
            raise JSONSchemaValidationError(
                "'total_records' must be an integer"
            )

        # Validate summary
        summary = data["summary"]
        missing_summary = self.REQUIRED_SUMMARY_KEYS - set(summary.keys())
        if missing_summary:
            raise JSONSchemaValidationError(
                f"Missing summary fields: {missing_summary}"
            )

        # Validate transactions is a list
        if not isinstance(data["transactions"], list):
            raise JSONSchemaValidationError("'transactions' must be a list")

        # Validate holders is a list
        if not isinstance(data["holders"], list):
            raise JSONSchemaValidationError("'holders' must be a list")

        # Validate each transaction has required fields
        tx_required_fields = {
            "transaction_hash", "block_number", "timestamp",
            "from_address", "to_address", "amount",
            "stablecoin", "chain", "activity_type", "source_explorer",
        }
        for i, tx in enumerate(data["transactions"]):
            missing_tx_fields = tx_required_fields - set(tx.keys())
            if missing_tx_fields:
                raise JSONSchemaValidationError(
                    f"Transaction {i} missing fields: {missing_tx_fields}"
                )

        # Validate each holder has required fields
        holder_required_fields = {
            "address", "balance", "stablecoin", "chain",
            "first_seen", "last_activity", "is_store_of_value",
            "source_explorer",
        }
        for i, holder in enumerate(data["holders"]):
            missing_holder_fields = holder_required_fields - set(holder.keys())
            if missing_holder_fields:
                raise JSONSchemaValidationError(
                    f"Holder {i} missing fields: {missing_holder_fields}"
                )

        logger.debug("JSON schema validation passed")
        return True

    def _ensure_output_directory(self) -> Path:
        """Ensure the output directory exists.

        Returns:
            Path object for the output directory.

        Raises:
            JSONExportError: If directory cannot be created.
        """
        output_path = Path(self._output_directory)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path
        except OSError as e:
            msg = f"Failed to create output directory " \
                  f"'{self._output_directory}': {e}"
            raise JSONExportError(msg) from e

    async def export(
        self,
        data: AggregatedData,
        run_id: str,
        output_path: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Export aggregated data to JSON file.

        Args:
            data: Aggregated data to export.
            run_id: The unique run identifier.
            output_path: Optional custom output path. If not provided,
                uses configured output directory with generated filename.
            user_id: Optional user ID who initiated the run.

        Returns:
            Path to the written JSON file.

        Raises:
            JSONExportError: If export fails.
            JSONSchemaValidationError: If data validation fails.
        """
        # Build output data structure
        output_data = self._build_output_data(data, run_id, user_id)

        # Validate schema before writing
        self.validate_json_schema(output_data)

        # Determine output file path
        if output_path:
            file_path = Path(output_path)
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self._ensure_output_directory()
            filename = self.generate_filename(run_id)
            file_path = output_dir / filename

        # Write JSON file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Exported {data.total_records} records to {file_path}",
                extra={
                    "run_id": run_id,
                    "file_path": str(file_path),
                    "total_records": data.total_records,
                    "explorers": data.explorers_queried,
                }
            )

            return str(file_path)

        except (OSError, IOError) as e:
            raise JSONExportError(
                f"Failed to write JSON file '{file_path}': {e}"
            ) from e

    async def save_to_database(
        self,
        run_id: str,
        data: AggregatedData,
        output_file_path: str
    ) -> str:
        """Save run results metadata to database.

        Args:
            run_id: The unique run identifier.
            data: Aggregated data that was exported.
            output_file_path: Path to the exported JSON file.

        Returns:
            The result_id as a string.

        Raises:
            JSONExportError: If database save fails or no db_manager
                configured.
        """
        if self._db_manager is None:
            raise JSONExportError(
                "Cannot save to database: no DatabaseManager configured"
            )

        # Build summary for database storage
        summary = {
            "by_stablecoin": {
                coin: summary.to_dict()
                for coin, summary in data.by_stablecoin.items()
            },
            "by_activity_type": data.by_activity_type,
            "by_chain": data.by_chain,
        }

        try:
            result_id = await self._db_manager.save_run_result(
                run_id=run_id,
                total_records=data.total_records,
                explorers_queried=data.explorers_queried,
                output_file_path=output_file_path,
                summary=summary
            )

            logger.info(
                f"Saved run result {result_id} to database",
                extra={
                    "run_id": run_id,
                    "result_id": result_id,
                    "total_records": data.total_records,
                }
            )

            return result_id

        except Exception as e:
            raise JSONExportError(
                f"Failed to save run result to database: {e}"
            ) from e

    async def export_and_save(
        self,
        data: AggregatedData,
        run_id: str,
        output_path: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """Export to JSON file and save to database in one operation.

        Convenience method that combines export() and save_to_database().

        Args:
            data: Aggregated data to export.
            run_id: The unique run identifier.
            output_path: Optional custom output path.
            user_id: Optional user ID who initiated the run.

        Returns:
            Tuple of (file_path, result_id). result_id is None if no
            database manager is configured.

        Raises:
            JSONExportError: If export or database save fails.
            JSONSchemaValidationError: If data validation fails.
        """
        # Export to file
        file_path = await self.export(data, run_id, output_path, user_id)

        # Save to database if manager is configured
        result_id = None
        if self._db_manager is not None:
            result_id = await self.save_to_database(run_id, data, file_path)

        return file_path, result_id
