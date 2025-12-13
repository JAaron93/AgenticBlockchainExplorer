"""Model performance monitoring for stablecoin analysis ML models.

This module provides functionality to:
- Check model metrics against configured thresholds
- Detect performance degradation relative to production models
- Generate alerts for degraded performance
- Block automatic promotion of underperforming models

Requirements: 15.4, 15.5
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_THRESHOLDS_PATH = Path("config/ml_thresholds.json")


@dataclass
class MetricThresholds:
    """Thresholds for model performance metrics.
    
    Attributes:
        precision_min: Minimum acceptable precision
        recall_min: Minimum acceptable recall
        f1_min: Minimum acceptable F1 score
        auc_min: Minimum acceptable AUC-ROC
        relative_drop_max: Maximum acceptable relative drop from production
    """
    precision_min: float = 0.70
    recall_min: float = 0.65
    f1_min: float = 0.67
    auc_min: float = 0.75
    relative_drop_max: float = 0.10
    
    @classmethod
    def from_dict(cls, data: dict) -> "MetricThresholds":
        """Create from dictionary."""
        return cls(
            precision_min=data.get("precision_min", 0.70),
            recall_min=data.get("recall_min", 0.65),
            f1_min=data.get("f1_min", 0.67),
            auc_min=data.get("auc_min", 0.75),
            relative_drop_max=data.get("relative_drop_max", 0.10),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "precision_min": self.precision_min,
            "recall_min": self.recall_min,
            "f1_min": self.f1_min,
            "auc_min": self.auc_min,
            "relative_drop_max": self.relative_drop_max,
        }


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring.
    
    Attributes:
        thresholds: Per-model metric thresholds
        alerts_enabled: Whether to send alerts
        alert_channels: Configured alert channels
        auto_promotion_enabled: Whether to auto-promote passing models
    """
    thresholds: Dict[str, MetricThresholds] = field(default_factory=dict)
    alerts_enabled: bool = True
    alert_channels: Dict[str, Any] = field(default_factory=dict)
    auto_promotion_enabled: bool = False
    require_all_thresholds: bool = True
    require_improvement: bool = True
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "MonitoringConfig":
        """Load configuration from JSON file."""
        config_path = path or DEFAULT_THRESHOLDS_PATH
        
        if not config_path.exists():
            logger.warning(
                f"Config file not found at {config_path}, using defaults"
            )
            return cls()
        
        try:
            with open(config_path) as f:
                data = json.load(f)
            
            thresholds = {}
            for model_name, thresh_data in data.get("thresholds", {}).items():
                thresholds[model_name] = MetricThresholds.from_dict(thresh_data)
            
            alerts = data.get("alerts", {})
            auto_promo = data.get("auto_promotion", {})
            
            return cls(
                thresholds=thresholds,
                alerts_enabled=alerts.get("enabled", True),
                alert_channels=alerts.get("channels", {}),
                auto_promotion_enabled=auto_promo.get("enabled", False),
                require_all_thresholds=auto_promo.get(
                    "require_all_thresholds", True
                ),
                require_improvement=auto_promo.get(
                    "require_improvement_over_production", True
                ),
            )
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()
    
    def get_thresholds(self, model_name: str) -> MetricThresholds:
        """Get thresholds for a specific model."""
        return self.thresholds.get(model_name, MetricThresholds())


# =============================================================================
# Monitoring Results
# =============================================================================

class DegradationSeverity(str, Enum):
    """Severity level for performance degradation."""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ThresholdViolation:
    """A single threshold violation.
    
    Attributes:
        metric_name: Name of the violated metric
        actual_value: Actual metric value
        threshold_value: Threshold that was violated
        violation_type: Type of violation (absolute or relative)
        severity: Severity of the violation
    """
    metric_name: str
    actual_value: float
    threshold_value: float
    violation_type: str  # "absolute" or "relative"
    severity: DegradationSeverity = DegradationSeverity.WARNING
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "actual_value": self.actual_value,
            "threshold_value": self.threshold_value,
            "violation_type": self.violation_type,
            "severity": self.severity.value,
        }


@dataclass
class MonitoringResult:
    """Result of model performance monitoring.
    
    Attributes:
        model_name: Name of the monitored model
        model_version: Version of the model
        metrics: Model metrics that were evaluated
        violations: List of threshold violations
        is_degraded: Whether the model shows degradation
        severity: Overall severity of degradation
        timestamp: When the check was performed
        production_metrics: Metrics of the production model (if compared)
        recommendation: Recommended action
    """
    model_name: str
    model_version: str
    metrics: Dict[str, float]
    violations: List[ThresholdViolation] = field(default_factory=list)
    is_degraded: bool = False
    severity: DegradationSeverity = DegradationSeverity.NONE
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    production_metrics: Optional[Dict[str, float]] = None
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "metrics": self.metrics,
            "violations": [v.to_dict() for v in self.violations],
            "is_degraded": self.is_degraded,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "production_metrics": self.production_metrics,
            "recommendation": self.recommendation,
        }


# =============================================================================
# Model Performance Monitor
# =============================================================================

class ModelPerformanceMonitor:
    """Monitor ML model performance and detect degradation.
    
    Checks model metrics against configured thresholds and compares
    with production model performance to detect degradation.
    
    Requirements: 15.4, 15.5
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize the monitor.
        
        Args:
            config: Monitoring configuration (loads from file if not provided)
        """
        self.config = config or MonitoringConfig.load()
        self._audit_log: List[Dict[str, Any]] = []
    
    def check_model(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float],
        production_metrics: Optional[Dict[str, float]] = None,
    ) -> MonitoringResult:
        """Check model metrics against thresholds.
        
        Args:
            model_name: Name of the model being checked
            model_version: Version of the model
            metrics: Model metrics (precision, recall, f1, auc)
            production_metrics: Optional metrics from production model
            
        Returns:
            MonitoringResult with violations and recommendations
            
        Requirements: 15.4, 15.5
        """
        thresholds = self.config.get_thresholds(model_name)
        violations = []
        
        # Check absolute thresholds
        absolute_checks = [
            ("precision", metrics.get("precision", 0), thresholds.precision_min),
            ("recall", metrics.get("recall", 0), thresholds.recall_min),
            ("f1", metrics.get("f1", 0), thresholds.f1_min),
            ("auc", metrics.get("auc", 0), thresholds.auc_min),
        ]
        
        for metric_name, actual, threshold in absolute_checks:
            if actual < threshold:
                severity = (
                    DegradationSeverity.CRITICAL 
                    if actual < threshold * 0.9 
                    else DegradationSeverity.WARNING
                )
                violations.append(ThresholdViolation(
                    metric_name=metric_name,
                    actual_value=actual,
                    threshold_value=threshold,
                    violation_type="absolute",
                    severity=severity,
                ))
        
        # Check relative degradation vs production
        if production_metrics:
            for metric_name in ["precision", "recall", "f1", "auc"]:
                prod_value = production_metrics.get(metric_name, 0)
                new_value = metrics.get(metric_name, 0)
                
                if prod_value > 0:
                    relative_drop = (prod_value - new_value) / prod_value
                    if relative_drop > thresholds.relative_drop_max:
                        violations.append(ThresholdViolation(
                            metric_name=metric_name,
                            actual_value=new_value,
                            threshold_value=prod_value,
                            violation_type="relative",
                            severity=DegradationSeverity.WARNING,
                        ))
        
        # Determine overall result
        is_degraded = len(violations) > 0
        severity = DegradationSeverity.NONE
        if violations:
            if any(v.severity == DegradationSeverity.CRITICAL for v in violations):
                severity = DegradationSeverity.CRITICAL
            else:
                severity = DegradationSeverity.WARNING
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            violations, is_degraded, severity
        )
        
        result = MonitoringResult(
            model_name=model_name,
            model_version=model_version,
            metrics=metrics,
            violations=violations,
            is_degraded=is_degraded,
            severity=severity,
            production_metrics=production_metrics,
            recommendation=recommendation,
        )
        
        # Log to audit trail
        self._log_check(result)
        
        # Send alerts if configured
        if is_degraded and self.config.alerts_enabled:
            self._send_alerts(result)
        
        return result
    
    def should_block_promotion(self, result: MonitoringResult) -> bool:
        """Determine if model promotion should be blocked.
        
        Args:
            result: Monitoring result from check_model
            
        Returns:
            True if promotion should be blocked
            
        Requirements: 15.5
        """
        if not result.is_degraded:
            return False
        
        # Block if any critical violations
        if result.severity == DegradationSeverity.CRITICAL:
            return True
        
        # Block if configured to require all thresholds
        if self.config.require_all_thresholds and result.violations:
            return True
        
        return False
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log of monitoring checks."""
        return self._audit_log.copy()
    
    def _generate_recommendation(
        self,
        violations: List[ThresholdViolation],
        is_degraded: bool,
        severity: DegradationSeverity,
    ) -> str:
        """Generate a recommendation based on violations."""
        if not is_degraded:
            return "Model meets all performance thresholds. Safe to promote."
        
        if severity == DegradationSeverity.CRITICAL:
            return (
                "CRITICAL: Model shows significant performance degradation. "
                "Do not promote. Review training data and model configuration."
            )
        
        violation_metrics = [v.metric_name for v in violations]
        return (
            f"WARNING: Model shows degradation in: {', '.join(violation_metrics)}. "
            "Manual review required before promotion."
        )
    
    def _log_check(self, result: MonitoringResult) -> None:
        """Log monitoring check to audit trail."""
        log_entry = {
            "timestamp": result.timestamp.isoformat(),
            "model_name": result.model_name,
            "model_version": result.model_version,
            "metrics": result.metrics,
            "is_degraded": result.is_degraded,
            "severity": result.severity.value,
            "violations_count": len(result.violations),
            "thresholds_violated": [v.metric_name for v in result.violations],
        }
        self._audit_log.append(log_entry)
        
        # Also log to standard logger
        if result.is_degraded:
            logger.warning(
                f"Model degradation detected: {result.model_name} "
                f"v{result.model_version} - {result.severity.value}"
            )
        else:
            logger.info(
                f"Model check passed: {result.model_name} "
                f"v{result.model_version}"
            )
    
    def _send_alerts(self, result: MonitoringResult) -> None:
        """Send alerts for degraded model performance."""
        channels = self.config.alert_channels
        
        # Log-only alert (always enabled as fallback)
        if channels.get("log_only", {}).get("enabled", True):
            logger.warning(
                f"MODEL DEGRADATION ALERT: {result.model_name} "
                f"v{result.model_version}\n"
                f"Severity: {result.severity.value}\n"
                f"Violations: {[v.to_dict() for v in result.violations]}\n"
                f"Recommendation: {result.recommendation}"
            )
        
        # Slack alert
        if channels.get("slack", {}).get("enabled", False):
            webhook_url = channels["slack"].get("webhook_url")
            if webhook_url:
                self._send_slack_alert(result, webhook_url)
        
        # Email alert
        if channels.get("email", {}).get("enabled", False):
            recipients = channels["email"].get("recipients", [])
            if recipients:
                self._send_email_alert(result, recipients)
    
    def _send_slack_alert(
        self, result: MonitoringResult, webhook_url: str
    ) -> None:
        """Send alert to Slack webhook."""
        try:
            import requests
            
            message = {
                "text": ":warning: Model Degradation Alert",
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"Model Degradation: {result.model_name}",
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Version:* {result.model_version}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Severity:* {result.severity.value}"
                            },
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Recommendation:* {result.recommendation}"
                        }
                    },
                ]
            }
            
            requests.post(webhook_url, json=message, timeout=10)
            logger.info("Slack alert sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_email_alert(
        self, result: MonitoringResult, recipients: List[str]
    ) -> None:
        """Send alert via email."""
        # Email sending would require SMTP configuration
        # For now, just log that we would send an email
        logger.info(
            f"Would send email alert to {recipients} for "
            f"{result.model_name} v{result.model_version}"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def check_model_performance(
    model_name: str,
    model_version: str,
    metrics: Dict[str, float],
    production_metrics: Optional[Dict[str, float]] = None,
    config_path: Optional[Path] = None,
) -> MonitoringResult:
    """Check model performance against thresholds.
    
    Convenience function for checking model performance.
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        metrics: Model metrics (precision, recall, f1, auc)
        production_metrics: Optional metrics from production model
        config_path: Optional path to config file
        
    Returns:
        MonitoringResult with violations and recommendations
        
    Requirements: 15.4, 15.5
    """
    config = MonitoringConfig.load(config_path)
    monitor = ModelPerformanceMonitor(config)
    return monitor.check_model(
        model_name=model_name,
        model_version=model_version,
        metrics=metrics,
        production_metrics=production_metrics,
    )


def is_model_degraded(
    model_name: str,
    metrics: Dict[str, float],
    config_path: Optional[Path] = None,
) -> bool:
    """Quick check if model shows degradation.
    
    Args:
        model_name: Name of the model
        metrics: Model metrics
        config_path: Optional path to config file
        
    Returns:
        True if model shows degradation
    """
    result = check_model_performance(
        model_name=model_name,
        model_version="check",
        metrics=metrics,
        config_path=config_path,
    )
    return result.is_degraded
