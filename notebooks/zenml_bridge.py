"""ZenML-Marimo bridge for pipeline control and artifact loading.

This module provides the ZenMLNotebookBridge class that enables marimo notebooks
to interact with ZenML pipelines:
- List available pipelines and their status
- Trigger pipeline runs with parameters
- Monitor pipeline run status
- Load artifacts from completed runs

Requirements: 13.1, 13.2, 13.3, 13.4
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Bridge Outputs
# =============================================================================

@dataclass
class PipelineInfo:
    """Information about a ZenML pipeline.
    
    Attributes:
        name: Pipeline name
        last_run_id: ID of the last run (if any)
        last_run_time: Timestamp of the last run
        last_run_status: Status of the last run
        total_runs: Total number of runs
    """
    name: str
    last_run_id: Optional[str] = None
    last_run_time: Optional[datetime] = None
    last_run_status: Optional[str] = None
    total_runs: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "last_run_id": self.last_run_id,
            "last_run_time": (
                self.last_run_time.isoformat() 
                if self.last_run_time else None
            ),
            "last_run_status": self.last_run_status,
            "total_runs": self.total_runs,
        }


@dataclass
class PipelineRunStatus:
    """Status of a pipeline run.
    
    Attributes:
        run_id: Unique run identifier
        pipeline_name: Name of the pipeline
        status: Overall run status (running, completed, failed)
        start_time: When the run started
        end_time: When the run ended (if completed)
        steps: Status of individual steps
        error_message: Error message if failed
    """
    run_id: str
    pipeline_name: str
    status: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    steps: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status,
            "start_time": (
                self.start_time.isoformat() 
                if self.start_time else None
            ),
            "end_time": (
                self.end_time.isoformat() 
                if self.end_time else None
            ),
            "steps": self.steps,
            "error_message": self.error_message,
        }
    
    @property
    def is_running(self) -> bool:
        """Check if the pipeline is still running."""
        return self.status in ("running", "initializing", "pending")
    
    @property
    def is_completed(self) -> bool:
        """Check if the pipeline completed successfully."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Check if the pipeline failed."""
        return self.status == "failed"


@dataclass
class LoadedArtifacts:
    """Artifacts loaded from a pipeline run.
    
    Attributes:
        run_id: ID of the run artifacts were loaded from
        run_timestamp: When the run was executed
        artifacts: Dictionary of artifact name to artifact data
        metadata: Additional metadata about the artifacts
    """
    run_id: str
    run_timestamp: Optional[datetime] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_timestamp": (
                self.run_timestamp.isoformat() 
                if self.run_timestamp else None
            ),
            "metadata": self.metadata,
            # Note: artifacts not serialized as they may be large DataFrames
        }


# =============================================================================
# ZenML Notebook Bridge
# =============================================================================

class ZenMLNotebookBridge:
    """Bridge between marimo notebook and ZenML pipelines.
    
    Provides methods to:
    - List available pipelines and their status
    - Trigger pipeline runs with parameters
    - Monitor pipeline run status
    - Load artifacts from completed runs
    
    Requirements: 13.1, 13.2, 13.3, 13.4
    """
    
    def __init__(self, force_mock: bool = False):
        """Initialize the ZenML bridge.
        
        Attempts to connect to ZenML. If ZenML is not available or
        not properly configured, operates in mock mode for development/testing.
        
        Args:
            force_mock: If True, always use mock mode regardless of ZenML availability
        """
        self._client = None
        self._mock_mode = force_mock
        self._mock_runs: Dict[str, PipelineRunStatus] = {}
        
        if force_mock:
            logger.info("ZenML bridge initialized in forced mock mode")
            return
        
        try:
            from zenml.client import Client
            self._client = Client()
            # Test that the client is actually usable by checking zen_store
            # This will fail if ZenML is not properly configured
            _ = self._client.active_stack
            logger.info("ZenML client initialized successfully")
        except ImportError:
            logger.warning("ZenML not installed, operating in mock mode")
            self._mock_mode = True
        except Exception as e:
            logger.warning(f"Failed to initialize ZenML client: {e}, operating in mock mode")
            self._mock_mode = True
            self._client = None
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to ZenML."""
        return self._client is not None and not self._mock_mode
    
    def list_pipelines(self) -> List[PipelineInfo]:
        """List available ZenML pipelines with status.
        
        Returns:
            List of PipelineInfo objects with pipeline details
            
        Requirements: 13.1
        """
        if self._mock_mode:
            return self._mock_list_pipelines()
        
        try:
            pipelines = self._client.list_pipelines()
            result = []
            
            for pipeline in pipelines:
                # Get the latest run for this pipeline
                runs = self._client.list_pipeline_runs(
                    pipeline_id=pipeline.id,
                    sort_by="desc:created",
                    size=1,
                )
                
                last_run = runs[0] if runs else None
                
                info = PipelineInfo(
                    name=pipeline.name,
                    last_run_id=str(last_run.id) if last_run else None,
                    last_run_time=last_run.created if last_run else None,
                    last_run_status=str(last_run.status) if last_run else None,
                    total_runs=len(self._client.list_pipeline_runs(
                        pipeline_id=pipeline.id
                    )),
                )
                result.append(info)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list pipelines: {e}")
            return []
    
    def trigger_pipeline(
        self,
        pipeline_name: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Trigger a pipeline run and return run_id.
        
        Args:
            pipeline_name: Name of the pipeline to run
            parameters: Optional parameters to pass to the pipeline
            
        Returns:
            run_id: Unique identifier for the triggered run
            
        Raises:
            ValueError: If pipeline not found
            RuntimeError: If pipeline trigger fails
            
        Requirements: 13.2
        """
        if parameters is None:
            parameters = {}
        
        if self._mock_mode:
            return self._mock_trigger_pipeline(pipeline_name, parameters)
        
        try:
            # Get the pipeline
            pipeline = self._client.get_pipeline(pipeline_name)
            if pipeline is None:
                raise ValueError(f"Pipeline '{pipeline_name}' not found")
            
            # Import and run the pipeline based on name
            if pipeline_name == "stablecoin_master_pipeline":
                from pipelines.master_pipeline import master_pipeline
                run = master_pipeline.with_options(
                    run_name=f"{pipeline_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                )(**parameters)
            elif pipeline_name == "stablecoin_collection_pipeline":
                from pipelines.collection_pipeline import collection_pipeline
                run = collection_pipeline.with_options(
                    run_name=f"{pipeline_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                )(**parameters)
            elif pipeline_name == "stablecoin_analysis_pipeline":
                from pipelines.analysis_pipeline import analysis_pipeline
                run = analysis_pipeline.with_options(
                    run_name=f"{pipeline_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
                )(**parameters)
            else:
                raise ValueError(f"Unknown pipeline: {pipeline_name}")
            
            run_id = str(run.id) if hasattr(run, 'id') else str(run)
            logger.info(f"Triggered pipeline '{pipeline_name}' with run_id: {run_id}")
            return run_id
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to trigger pipeline '{pipeline_name}': {e}")
            raise RuntimeError(f"Failed to trigger pipeline: {e}")
    
    def get_run_status(self, run_id: str) -> PipelineRunStatus:
        """Get status of a pipeline run.
        
        Args:
            run_id: Unique identifier of the pipeline run
            
        Returns:
            PipelineRunStatus with current status and step details
            
        Requirements: 13.3
        """
        if self._mock_mode:
            return self._mock_get_run_status(run_id)
        
        try:
            run = self._client.get_pipeline_run(run_id)
            
            # Get step statuses
            steps = {}
            for step_name, step in run.steps.items():
                steps[step_name] = str(step.status)
            
            return PipelineRunStatus(
                run_id=str(run.id),
                pipeline_name=run.pipeline.name if run.pipeline else "unknown",
                status=str(run.status),
                start_time=run.created,
                end_time=run.end_time if hasattr(run, 'end_time') else None,
                steps=steps,
                error_message=None,  # Would need to extract from failed steps
            )
            
        except Exception as e:
            logger.error(f"Failed to get run status for '{run_id}': {e}")
            return PipelineRunStatus(
                run_id=run_id,
                pipeline_name="unknown",
                status="error",
                error_message=str(e),
            )
    
    def load_latest_artifacts(
        self,
        pipeline_name: str = "stablecoin_master_pipeline",
    ) -> Optional[LoadedArtifacts]:
        """Load artifacts from the latest successful pipeline run.
        
        Args:
            pipeline_name: Name of the pipeline to load artifacts from
            
        Returns:
            LoadedArtifacts with data from the latest run, or None if no runs
            
        Requirements: 13.4
        """
        if self._mock_mode:
            return self._mock_load_latest_artifacts(pipeline_name)
        
        try:
            # Get the latest successful run
            runs = self._client.list_pipeline_runs(
                pipeline_name=pipeline_name,
                status="completed",
                sort_by="desc:created",
                size=1,
            )
            
            if not runs:
                logger.warning(f"No completed runs found for pipeline '{pipeline_name}'")
                return None
            
            latest_run = runs[0]
            
            # Load artifacts from each step
            artifacts = {}
            
            # Map step names to artifact names
            step_artifact_map = {
                "activity_analysis_step": "activity_breakdown",
                "holder_analysis_step": "holder_metrics",
                "time_series_step": "time_series",
                "chain_analysis_step": "chain_metrics",
                "predict_sov_step": "sov_predictions",
                "classify_wallets_step": "wallet_classifications",
            }
            
            for step_name, artifact_name in step_artifact_map.items():
                if step_name in latest_run.steps:
                    step = latest_run.steps[step_name]
                    if step.outputs:
                        # Load the first output artifact
                        for output_name, artifact in step.outputs.items():
                            try:
                                artifacts[artifact_name] = artifact.load()
                                break
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load artifact '{output_name}' "
                                    f"from step '{step_name}': {e}"
                                )
            
            return LoadedArtifacts(
                run_id=str(latest_run.id),
                run_timestamp=latest_run.created,
                artifacts=artifacts,
                metadata={
                    "pipeline_name": pipeline_name,
                    "loaded_at": datetime.now(timezone.utc).isoformat(),
                    "artifact_count": len(artifacts),
                },
            )
            
        except Exception as e:
            logger.error(f"Failed to load artifacts from '{pipeline_name}': {e}")
            return None
    
    def get_model_versions(
        self,
        model_name: str,
    ) -> List[Dict[str, Any]]:
        """List all versions of a model with metrics.
        
        Args:
            model_name: Name of the model to list versions for
            
        Returns:
            List of model version info dictionaries
        """
        if self._mock_mode:
            return []
        
        try:
            versions = self._client.list_model_versions(model_name)
            return [
                {
                    "version": str(v.version),
                    "created": v.created.isoformat() if v.created else None,
                    "metrics": v.metadata.get("metrics", {}) if v.metadata else {},
                    "is_production": v.stage == "production",
                }
                for v in versions
            ]
        except Exception as e:
            logger.error(f"Failed to get model versions for '{model_name}': {e}")
            return []
    
    def promote_model(
        self,
        model_name: str,
        version: str,
    ) -> bool:
        """Promote a model version to production.
        
        Args:
            model_name: Name of the model
            version: Version to promote
            
        Returns:
            True if promotion succeeded, False otherwise
        """
        if self._mock_mode:
            return True
        
        try:
            self._client.update_model_version(
                model_name,
                version,
                stage="production",
            )
            logger.info(f"Promoted model '{model_name}' version '{version}' to production")
            return True
        except Exception as e:
            logger.error(f"Failed to promote model '{model_name}' version '{version}': {e}")
            return False
    
    # =========================================================================
    # Mock Mode Methods (for development/testing)
    # =========================================================================
    
    def _mock_list_pipelines(self) -> List[PipelineInfo]:
        """Mock implementation of list_pipelines."""
        return [
            PipelineInfo(
                name="stablecoin_master_pipeline",
                last_run_id="mock-run-001",
                last_run_time=datetime.now(timezone.utc),
                last_run_status="completed",
                total_runs=5,
            ),
            PipelineInfo(
                name="stablecoin_collection_pipeline",
                last_run_id="mock-run-002",
                last_run_time=datetime.now(timezone.utc),
                last_run_status="completed",
                total_runs=10,
            ),
            PipelineInfo(
                name="stablecoin_analysis_pipeline",
                last_run_id="mock-run-003",
                last_run_time=datetime.now(timezone.utc),
                last_run_status="completed",
                total_runs=8,
            ),
        ]
    
    def _mock_trigger_pipeline(
        self,
        pipeline_name: str,
        parameters: Dict[str, Any],
    ) -> str:
        """Mock implementation of trigger_pipeline."""
        import uuid
        run_id = f"mock-{uuid.uuid4().hex[:8]}"
        
        # Store mock run status
        self._mock_runs[run_id] = PipelineRunStatus(
            run_id=run_id,
            pipeline_name=pipeline_name,
            status="completed",  # Mock runs complete immediately
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            steps={
                "etherscan_collector_step": "completed",
                "bscscan_collector_step": "completed",
                "polygonscan_collector_step": "completed",
                "aggregate_data_step": "completed",
                "activity_analysis_step": "completed",
                "holder_analysis_step": "completed",
                "time_series_step": "completed",
                "chain_analysis_step": "completed",
            },
        )
        
        logger.info(f"[MOCK] Triggered pipeline '{pipeline_name}' with run_id: {run_id}")
        return run_id
    
    def _mock_get_run_status(self, run_id: str) -> PipelineRunStatus:
        """Mock implementation of get_run_status."""
        if run_id in self._mock_runs:
            return self._mock_runs[run_id]
        
        # Return a default completed status for unknown runs
        return PipelineRunStatus(
            run_id=run_id,
            pipeline_name="unknown",
            status="completed",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            steps={},
        )
    
    def _mock_load_latest_artifacts(
        self,
        pipeline_name: str,
    ) -> Optional[LoadedArtifacts]:
        """Mock implementation of load_latest_artifacts."""
        import pandas as pd
        
        # Return mock artifacts
        return LoadedArtifacts(
            run_id="mock-latest-run",
            run_timestamp=datetime.now(timezone.utc),
            artifacts={
                "activity_breakdown": {
                    "counts": {"transaction": 500, "store_of_value": 300, "other": 200},
                    "percentages": {"transaction": 50.0, "store_of_value": 30.0, "other": 20.0},
                },
                "holder_metrics": {
                    "total_holders": 100,
                    "sov_count": 30,
                    "sov_percentage": 30.0,
                },
                "time_series": pd.DataFrame({
                    "period": pd.date_range("2024-01-01", periods=7, freq="D"),
                    "transaction_count": [100, 120, 90, 110, 130, 95, 105],
                }),
                "chain_metrics": {
                    "chain_metrics": [
                        {"chain": "ethereum", "transaction_count": 400},
                        {"chain": "bsc", "transaction_count": 350},
                        {"chain": "polygon", "transaction_count": 250},
                    ]
                },
            },
            metadata={
                "pipeline_name": pipeline_name,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "is_mock": True,
            },
        )
