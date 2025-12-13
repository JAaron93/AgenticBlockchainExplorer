# Pipeline Scheduling Guide

This document describes how to configure and run scheduled pipelines for the stablecoin analysis system.

## Overview

The stablecoin analysis system supports scheduled execution of pipelines via:
1. **ZenML Scheduling** - Native ZenML schedule support
2. **System Cron** - Traditional cron jobs for environments without ZenML scheduling
3. **External Schedulers** - Integration with Airflow, Prefect, or other orchestrators

## Configuration

### Scheduling Configuration File

The scheduling configuration is stored in `config/scheduling.json`. This file defines:
- Pipeline schedules with cron expressions
- Pipeline parameters for each scheduled run
- Notification settings
- Retry policies

**Example config/scheduling.json:**

```json
{
  "schedules": [
    {
      "name": "weekly_master_pipeline",
      "pipeline": "master_pipeline",
      "cron_expression": "0 0 * * 0",
      "parameters": {
        "stablecoins": ["USDC", "USDT"],
        "date_range_days": 7,
        "max_records": 1000,
        "min_successful_collectors": 2,
        "aggregation": "daily",
        "top_n_holders": 10,
        "run_ml_inference": true
      },
      "notifications": {
        "on_failure": ["ops-team@example.com"],
        "on_success": false
      },
      "retry_policy": {
        "max_retries": 3,
        "backoff_multiplier": 2.0
      }
    },
    {
      "name": "daily_collection",
      "pipeline": "collection_pipeline",
      "cron_expression": "0 */6 * * *",
      "parameters": {
        "stablecoins": ["USDC", "USDT"],
        "date_range_days": 1
      },
      "notifications": {
        "on_failure": ["alerts@example.com"]
      },
      "retry_policy": {
        "max_retries": 2,
        "backoff_multiplier": 1.5
      }
    }
  ]
}
```

### Cron Expression Reference

| Expression | Description |
|------------|-------------|
| `0 0 * * 0` | Every Sunday at midnight UTC |
| `0 0 * * *` | Every day at midnight UTC |
| `0 */6 * * *` | Every 6 hours |
| `0 0 1 * *` | First day of each month at midnight |

## ZenML Scheduling

### Enable ZenML Scheduling

ZenML supports native scheduling when using orchestrators that support it (e.g., Kubeflow, Airflow).

```python
from pipelines.master_pipeline import run_weekly_master_pipeline

# Run with weekly schedule
run_weekly_master_pipeline(
    stablecoins=["USDC", "USDT"],
    date_range_days=7,
    cron_expression="0 0 * * 0",  # Every Sunday at midnight
)
```

### Check Schedule Status

```python
from zenml.client import Client

client = Client()
schedules = client.list_schedules()
for schedule in schedules:
    print(f"{schedule.name}: {schedule.cron_expression} - {schedule.active}")
```

## System Cron Setup

For environments without ZenML scheduling support, use system cron.

### Prerequisites

1. Python environment with all dependencies installed
2. ZenML initialized and configured
3. Environment variables set (API keys, etc.)

### Create Cron Script

Create a script at `scripts/run_master_pipeline.sh`:

```bash
#!/bin/bash
# Stablecoin Analysis Master Pipeline - Weekly Cron Job
# Schedule: Every Sunday at midnight UTC

set -e

# Configuration
PROJECT_DIR="/path/to/project"
VENV_DIR="${PROJECT_DIR}/venv"
LOG_DIR="${PROJECT_DIR}/logs"
LOG_FILE="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Change to project directory
cd "${PROJECT_DIR}"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Run the pipeline
echo "Starting master pipeline at $(date)" >> "${LOG_FILE}"
python -c "
from pipelines.master_pipeline import run_master_pipeline
from datetime import datetime

print(f'Pipeline started at {datetime.utcnow().isoformat()}')
result = run_master_pipeline(
    stablecoins=['USDC', 'USDT'],
    date_range_days=7,
    max_records=1000,
    min_successful_collectors=2,
    aggregation='daily',
    top_n_holders=10,
    run_ml_inference=True,
)
print(f'Pipeline completed at {datetime.utcnow().isoformat()}')
" >> "${LOG_FILE}" 2>&1

# Check exit status
if [ $? -eq 0 ]; then
    echo "Pipeline completed successfully at $(date)" >> "${LOG_FILE}"
else
    echo "Pipeline failed at $(date)" >> "${LOG_FILE}"
    # Send notification (optional)
    # curl -X POST -H 'Content-type: application/json' \
    #   --data '{"text":"Stablecoin pipeline failed!"}' \
    #   "${SLACK_WEBHOOK_URL}"
    exit 1
fi
```

### Install Cron Job

```bash
# Make script executable
chmod +x scripts/run_master_pipeline.sh

# Edit crontab
crontab -e

# Add the following line for weekly execution (Sunday midnight UTC):
0 0 * * 0 /path/to/project/scripts/run_master_pipeline.sh
```

### Verify Cron Job

```bash
# List current cron jobs
crontab -l

# Check cron logs (Linux)
grep CRON /var/log/syslog

# Check cron logs (macOS)
log show --predicate 'process == "cron"' --last 1h
```

## Web Export Scheduling

To automatically export results to the web after each pipeline run:

```python
from pipelines.master_pipeline import run_master_pipeline
from notebooks.web_exporter import export_analysis_to_web
from datetime import datetime, timezone
import uuid

# Run pipeline
result = run_master_pipeline()

# Export to web
run_id = str(uuid.uuid4())
export_analysis_to_web(
    run_id=run_id,
    analysis_results={
        "activity_breakdown": result.activity_breakdown.to_dict(),
        "holder_metrics": result.holder_metrics.to_dict(),
        "chain_metrics": result.chain_metrics.to_dict(),
    },
    predictions={
        "sov_predictions": result.sov_predictions.to_dict() if result.sov_predictions else None,
        "wallet_classifications": result.wallet_classifications.to_dict() if result.wallet_classifications else None,
    },
    data_timestamp=datetime.now(timezone.utc),
    data_completeness=result.run_metadata.get("data_completeness"),
)
```

## Monitoring

### Pipeline Run History

```python
from zenml.client import Client

client = Client()
runs = client.list_pipeline_runs(
    pipeline_name="stablecoin_master_pipeline",
    sort_by="desc:created",
    size=10,
)

for run in runs:
    print(f"{run.id}: {run.status} - {run.created}")
```

### Log Rotation

Configure log rotation to prevent disk space issues:

```bash
# /etc/logrotate.d/stablecoin-pipeline
/path/to/project/logs/*.log {
    weekly
    rotate 12
    compress
    delaycompress
    missingok
    notifempty
    create 644 user group
}
```

## Troubleshooting

### Common Issues

1. **Pipeline fails to start**
   - Check environment variables are set
   - Verify ZenML is initialized: `zenml status`
   - Check API keys are valid

2. **Cron job not running**
   - Verify cron service is running: `systemctl status cron`
   - Check cron syntax: `crontab -l`
   - Review cron logs for errors

3. **Pipeline times out**
   - Increase `run_timeout_hours` in scheduling config
   - Check network connectivity to blockchain APIs
   - Reduce `max_records` parameter

### Debug Mode

Run pipeline with debug logging:

```bash
export LOG_LEVEL=DEBUG
python -c "from pipelines.master_pipeline import run_master_pipeline; run_master_pipeline()"
```

## Requirements Reference

This scheduling implementation satisfies:
- **Requirement 10.5**: Weekly cron execution via ZenML's scheduling capabilities or external scheduler integration
