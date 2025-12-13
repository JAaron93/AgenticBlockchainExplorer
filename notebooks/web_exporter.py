"""Web export functionality for stablecoin analysis notebook.

This module provides static HTML/JSON export capabilities for the stablecoin
analysis notebook, enabling deployment to a live website.

Requirements: 14.5
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Export Configuration
# =============================================================================

@dataclass
class ExportConfig:
    """Configuration for web exports.
    
    Attributes:
        export_dir: Base directory for exports
        create_latest_symlink: Whether to create a 'latest' symlink
        include_visualizations: Whether to export HTML visualizations
        include_predictions: Whether to export ML predictions
        max_file_size_kb: Maximum size for individual HTML files
        max_total_size_mb: Maximum total export size
    """
    export_dir: Path = field(default_factory=lambda: Path("exports"))
    create_latest_symlink: bool = True
    include_visualizations: bool = True
    include_predictions: bool = True
    max_file_size_kb: int = 500
    max_total_size_mb: int = 10


@dataclass
class ExportManifest:
    """Manifest for an export run.
    
    Attributes:
        export_version: Schema version for the export
        run_id: Pipeline run identifier
        pipeline_name: Name of the pipeline that generated the data
        export_timestamp: When the export was created
        data_timestamp: When the data was collected
        files: List of exported files with metadata
        data_completeness: Information about data source completeness
        model_versions: Versions of ML models used
    """
    export_version: str = "1.0"
    run_id: str = ""
    pipeline_name: str = "master_pipeline"
    export_timestamp: str = ""
    data_timestamp: str = ""
    files: List[Dict[str, Any]] = field(default_factory=list)
    data_completeness: Dict[str, Any] = field(default_factory=dict)
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "export_version": self.export_version,
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "export_timestamp": self.export_timestamp,
            "data_timestamp": self.data_timestamp,
            "files": self.files,
            "data_completeness": self.data_completeness,
            "model_versions": self.model_versions,
        }


# =============================================================================
# JSON Encoder for Complex Types
# =============================================================================

class AnalysisJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for analysis data types."""
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


# =============================================================================
# Web Exporter Class
# =============================================================================

class WebExporter:
    """Export analysis results to static HTML/JSON for web deployment.
    
    Generates a complete export package including:
    - JSON analysis data files
    - HTML visualization files (standalone Altair charts)
    - Summary dashboard HTML
    - Manifest file with metadata
    
    Requirements: 14.5
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize the web exporter.
        
        Args:
            config: Export configuration (uses defaults if not provided)
        """
        self.config = config or ExportConfig()
        self._total_size = 0
    
    def export(
        self,
        run_id: str,
        analysis_results: Dict[str, Any],
        predictions: Optional[Dict[str, Any]] = None,
        visualizations: Optional[Dict[str, alt.Chart]] = None,
        data_timestamp: Optional[datetime] = None,
        data_completeness: Optional[Dict[str, Any]] = None,
        model_versions: Optional[Dict[str, str]] = None,
    ) -> Path:
        """Export analysis results to static files.
        
        Args:
            run_id: Unique identifier for this export
            analysis_results: Dictionary of analysis results
            predictions: Optional ML predictions to export
            visualizations: Optional Altair charts to export as HTML
            data_timestamp: When the data was collected
            data_completeness: Information about data source completeness
            model_versions: Versions of ML models used
            
        Returns:
            Path to the export directory
            
        Raises:
            ValueError: If export exceeds size limits
        """
        self._total_size = 0
        export_timestamp = datetime.now(timezone.utc)
        
        # Create export directory structure
        export_path = self.config.export_dir / run_id
        self._create_directory_structure(export_path)
        
        # Initialize manifest
        manifest = ExportManifest(
            run_id=run_id,
            export_timestamp=export_timestamp.isoformat(),
            data_timestamp=(
                data_timestamp.isoformat() 
                if data_timestamp else export_timestamp.isoformat()
            ),
            data_completeness=data_completeness or {},
            model_versions=model_versions or {},
        )
        
        # Export analysis JSON files
        self._export_analysis_files(export_path, analysis_results, manifest)
        
        # Export predictions if provided
        if predictions and self.config.include_predictions:
            self._export_predictions(export_path, predictions, manifest)
        
        # Export visualizations if provided
        if visualizations and self.config.include_visualizations:
            self._export_visualizations(export_path, visualizations, manifest)
        
        # Generate summary dashboard
        self._export_dashboard(
            export_path, 
            analysis_results, 
            predictions,
            run_id,
            data_timestamp or export_timestamp,
            data_completeness,
            manifest,
        )
        
        # Write manifest
        self._write_manifest(export_path, manifest)
        
        # Create latest symlink if configured
        if self.config.create_latest_symlink:
            self._create_latest_symlink(run_id)
        
        logger.info(
            f"Export completed: {export_path} "
            f"({self._total_size / 1024:.1f} KB total)"
        )
        
        return export_path
    
    def _create_directory_structure(self, export_path: Path) -> None:
        """Create the export directory structure."""
        directories = [
            export_path / "analysis",
            export_path / "predictions",
            export_path / "visualizations",
            export_path / "summary",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _export_analysis_files(
        self,
        export_path: Path,
        analysis_results: Dict[str, Any],
        manifest: ExportManifest,
    ) -> None:
        """Export analysis results as JSON files."""
        analysis_dir = export_path / "analysis"
        
        # Map of analysis types to file names
        analysis_files = {
            "activity_breakdown": "activity_breakdown.json",
            "stablecoin_comparison": "stablecoin_comparison.json",
            "holder_metrics": "holder_metrics.json",
            "time_series": "time_series.json",
            "chain_metrics": "chain_metrics.json",
        }
        
        for key, filename in analysis_files.items():
            if key in analysis_results:
                data = self._wrap_analysis_data(
                    analysis_results[key],
                    manifest.run_id,
                )
                file_path = analysis_dir / filename
                self._write_json_file(file_path, data, manifest)
    
    def _wrap_analysis_data(
        self,
        data: Any,
        run_id: str,
    ) -> dict:
        """Wrap analysis data with schema metadata."""
        return {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "data": data,
            "metadata": {},
        }
    
    def _export_predictions(
        self,
        export_path: Path,
        predictions: Dict[str, Any],
        manifest: ExportManifest,
    ) -> None:
        """Export ML predictions as JSON files."""
        predictions_dir = export_path / "predictions"
        
        prediction_files = {
            "sov_predictions": "sov_predictions.json",
            "wallet_classifications": "wallet_classifications.json",
        }
        
        for key, filename in prediction_files.items():
            if key in predictions:
                data = self._wrap_prediction_data(
                    predictions[key],
                    key,
                    manifest.model_versions.get(key, "unknown"),
                )
                file_path = predictions_dir / filename
                self._write_json_file(file_path, data, manifest)
    
    def _wrap_prediction_data(
        self,
        data: Any,
        model_name: str,
        model_version: str,
    ) -> dict:
        """Wrap prediction data with model metadata."""
        return {
            "schema_version": "1.0",
            "model_name": model_name,
            "model_version": model_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": data,
        }
    
    def _export_visualizations(
        self,
        export_path: Path,
        visualizations: Dict[str, alt.Chart],
        manifest: ExportManifest,
    ) -> None:
        """Export Altair charts as standalone HTML files."""
        viz_dir = export_path / "visualizations"
        
        for name, chart in visualizations.items():
            if chart is not None:
                filename = f"{name}.html"
                file_path = viz_dir / filename
                self._write_chart_html(file_path, chart, name, manifest)
    
    def _write_chart_html(
        self,
        file_path: Path,
        chart: alt.Chart,
        title: str,
        manifest: ExportManifest,
    ) -> None:
        """Write an Altair chart as standalone HTML."""
        try:
            # Generate HTML with embedded Vega-Lite spec
            html_content = self._generate_chart_html(chart, title)
            
            # Check size limit
            size_kb = len(html_content.encode('utf-8')) / 1024
            if size_kb > self.config.max_file_size_kb:
                logger.warning(
                    f"Chart {title} exceeds size limit "
                    f"({size_kb:.1f} KB > {self.config.max_file_size_kb} KB)"
                )
            
            # Write file
            file_path.write_text(html_content, encoding='utf-8')
            
            # Update manifest
            file_info = self._get_file_info(file_path, "html")
            manifest.files.append(file_info)
            self._total_size += file_info["size_bytes"]
            
        except Exception as e:
            logger.error(f"Failed to export chart {title}: {e}")
    
    def _generate_chart_html(self, chart: alt.Chart, title: str) -> str:
        """Generate standalone HTML for an Altair chart."""
        # Get the Vega-Lite spec
        spec = chart.to_dict()
        spec_json = json.dumps(spec)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Stablecoin Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            min-width: 320px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            font-size: 1.5rem;
            margin-bottom: 20px;
        }}
        #chart {{
            width: 100%;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div id="chart" role="img" aria-label="{title} visualization"></div>
    </div>
    <script>
        vegaEmbed('#chart', {spec_json}, {{
            actions: false,
            renderer: 'svg'
        }}).catch(console.error);
    </script>
</body>
</html>'''
        return html
    
    def _export_dashboard(
        self,
        export_path: Path,
        analysis_results: Dict[str, Any],
        predictions: Optional[Dict[str, Any]],
        run_id: str,
        data_timestamp: datetime,
        data_completeness: Optional[Dict[str, Any]],
        manifest: ExportManifest,
    ) -> None:
        """Generate the summary dashboard HTML."""
        summary_dir = export_path / "summary"
        
        # Export conclusions JSON
        conclusions = self._generate_conclusions(analysis_results)
        conclusions_path = summary_dir / "conclusions.json"
        self._write_json_file(conclusions_path, conclusions, manifest)
        
        # Generate dashboard HTML
        dashboard_html = self._generate_dashboard_html(
            analysis_results,
            predictions,
            conclusions,
            run_id,
            data_timestamp,
            data_completeness,
        )
        
        dashboard_path = summary_dir / "dashboard.html"
        dashboard_path.write_text(dashboard_html, encoding='utf-8')
        
        file_info = self._get_file_info(dashboard_path, "html")
        manifest.files.append(file_info)
        self._total_size += file_info["size_bytes"]
    
    def _generate_conclusions(
        self,
        analysis_results: Dict[str, Any],
    ) -> dict:
        """Generate conclusions from analysis results."""
        conclusions = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "findings": [],
        }
        
        # Extract key findings from analysis results
        if "activity_breakdown" in analysis_results:
            breakdown = analysis_results["activity_breakdown"]
            if isinstance(breakdown, dict):
                percentages = breakdown.get("percentages", {})
                dominant = max(percentages.items(), key=lambda x: x[1], default=(None, 0))
                if dominant[0]:
                    conclusions["findings"].append({
                        "type": "dominant_activity",
                        "value": dominant[0],
                        "percentage": dominant[1],
                        "description": f"Primary usage pattern: {dominant[0]} ({dominant[1]:.1f}%)",
                    })
        
        if "holder_metrics" in analysis_results:
            metrics = analysis_results["holder_metrics"]
            if isinstance(metrics, dict):
                sov_pct = metrics.get("sov_percentage", 0)
                conclusions["findings"].append({
                    "type": "sov_ratio",
                    "value": sov_pct,
                    "description": f"Store of Value holders: {sov_pct:.1f}%",
                })
        
        return conclusions
    
    def _generate_dashboard_html(
        self,
        analysis_results: Dict[str, Any],
        predictions: Optional[Dict[str, Any]],
        conclusions: dict,
        run_id: str,
        data_timestamp: datetime,
        data_completeness: Optional[Dict[str, Any]],
    ) -> str:
        """Generate the combined summary dashboard HTML."""
        completeness_ratio = 100.0
        if data_completeness:
            completeness_ratio = data_completeness.get("completeness_ratio", 1.0) * 100
        
        findings_html = ""
        for finding in conclusions.get("findings", []):
            findings_html += f'''
            <div class="finding">
                <span class="finding-type">{finding.get("type", "")}</span>
                <p>{finding.get("description", "")}</p>
            </div>'''
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stablecoin Analysis Dashboard - {run_id[:8]}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: #333;
        }}
        header {{
            background: linear-gradient(135deg, #2775CA, #26A17B);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        header h1 {{
            margin: 0 0 10px 0;
            font-size: 2rem;
        }}
        .timestamp {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}
        .completeness {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            margin-top: 10px;
        }}
        main {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        section h2 {{
            margin-top: 0;
            color: #2775CA;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .finding {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 10px;
            border-left: 4px solid #2775CA;
        }}
        .finding-type {{
            font-size: 0.8rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .finding p {{
            margin: 5px 0 0 0;
            font-size: 1.1rem;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .chart-embed {{
            border: 1px solid #eee;
            border-radius: 6px;
            overflow: hidden;
        }}
        .chart-embed iframe {{
            width: 100%;
            height: 350px;
            border: none;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }}
        @media (max-width: 600px) {{
            header h1 {{ font-size: 1.5rem; }}
            main {{ padding: 10px; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Stablecoin Usage Analysis</h1>
        <p class="timestamp">Data as of: {data_timestamp.strftime("%Y-%m-%d %H:%M UTC")}</p>
        <p class="completeness" role="status">Data completeness: {completeness_ratio:.0f}%</p>
    </header>
    <main>
        <section id="summary" aria-label="Key Findings">
            <h2>Key Findings</h2>
            {findings_html if findings_html else '<p>No findings available.</p>'}
        </section>
        <section id="activity" aria-label="Activity Analysis">
            <h2>Activity Analysis</h2>
            <div class="chart-grid">
                <div class="chart-embed">
                    <iframe src="../visualizations/activity_pie.html" title="Activity Distribution"></iframe>
                </div>
                <div class="chart-embed">
                    <iframe src="../visualizations/stablecoin_bar.html" title="Stablecoin Comparison"></iframe>
                </div>
            </div>
        </section>
        <section id="predictions" aria-label="ML Predictions">
            <h2>ML Predictions</h2>
            <div class="chart-grid">
                <div class="chart-embed">
                    <iframe src="../visualizations/sov_distribution.html" title="SoV Predictions"></iframe>
                </div>
                <div class="chart-embed">
                    <iframe src="../visualizations/wallet_class_breakdown.html" title="Wallet Classifications"></iframe>
                </div>
            </div>
        </section>
    </main>
    <footer>
        <p>Generated by Stablecoin Analysis Pipeline</p>
        <p>Run ID: {run_id}</p>
    </footer>
</body>
</html>'''
        return html
    
    def _write_json_file(
        self,
        file_path: Path,
        data: Any,
        manifest: ExportManifest,
    ) -> None:
        """Write data to a JSON file and update manifest."""
        content = json.dumps(data, cls=AnalysisJSONEncoder, indent=2)
        file_path.write_text(content, encoding='utf-8')
        
        file_info = self._get_file_info(file_path, "json")
        manifest.files.append(file_info)
        self._total_size += file_info["size_bytes"]
        
        # Check total size limit
        max_bytes = self.config.max_total_size_mb * 1024 * 1024
        if self._total_size > max_bytes:
            raise ValueError(
                f"Export exceeds maximum size limit "
                f"({self._total_size / 1024 / 1024:.1f} MB > "
                f"{self.config.max_total_size_mb} MB)"
            )
    
    def _write_manifest(
        self,
        export_path: Path,
        manifest: ExportManifest,
    ) -> None:
        """Write the export manifest file."""
        manifest_path = export_path / "manifest.json"
        content = json.dumps(manifest.to_dict(), indent=2)
        manifest_path.write_text(content, encoding='utf-8')
    
    def _get_file_info(self, file_path: Path, file_type: str) -> dict:
        """Get file metadata for manifest."""
        content = file_path.read_bytes()
        return {
            "path": str(file_path.relative_to(file_path.parent.parent)),
            "type": file_type,
            "size_bytes": len(content),
            "checksum_sha256": hashlib.sha256(content).hexdigest(),
        }
    
    def _create_latest_symlink(self, run_id: str) -> None:
        """Create or update the 'latest' symlink."""
        latest_path = self.config.export_dir / "latest"
        
        try:
            # Remove existing symlink if present
            if latest_path.is_symlink():
                latest_path.unlink()
            elif latest_path.exists():
                # If it's a regular file/directory, don't overwrite
                logger.warning(
                    f"Cannot create 'latest' symlink: "
                    f"{latest_path} exists and is not a symlink"
                )
                return
            
            # Create relative symlink
            latest_path.symlink_to(run_id)
            logger.info(f"Created 'latest' symlink -> {run_id}")
            
        except OSError as e:
            logger.warning(f"Failed to create 'latest' symlink: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def export_analysis_to_web(
    run_id: str,
    analysis_results: Dict[str, Any],
    predictions: Optional[Dict[str, Any]] = None,
    visualizations: Optional[Dict[str, alt.Chart]] = None,
    export_dir: Optional[Path] = None,
    data_timestamp: Optional[datetime] = None,
    data_completeness: Optional[Dict[str, Any]] = None,
    model_versions: Optional[Dict[str, str]] = None,
) -> Path:
    """Export analysis results to static web files.
    
    Convenience function for exporting analysis results.
    
    Args:
        run_id: Unique identifier for this export
        analysis_results: Dictionary of analysis results
        predictions: Optional ML predictions to export
        visualizations: Optional Altair charts to export as HTML
        export_dir: Directory for exports (default: exports/)
        data_timestamp: When the data was collected
        data_completeness: Information about data source completeness
        model_versions: Versions of ML models used
        
    Returns:
        Path to the export directory
        
    Requirements: 14.5
    """
    config = ExportConfig()
    if export_dir:
        config.export_dir = Path(export_dir)
    
    exporter = WebExporter(config)
    return exporter.export(
        run_id=run_id,
        analysis_results=analysis_results,
        predictions=predictions,
        visualizations=visualizations,
        data_timestamp=data_timestamp,
        data_completeness=data_completeness,
        model_versions=model_versions,
    )
