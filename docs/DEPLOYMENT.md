# Stablecoin Analysis Notebook - Deployment Guide

This guide covers deploying the stablecoin analysis notebook to a live website with scheduled weekly updates.

## Prerequisites

- Python 3.9+
- ZenML installed and configured
- Access to blockchain explorer APIs (Etherscan, BscScan, Polygonscan)
- Web server for hosting static exports

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Cron Job   │───▶│ ZenML Master │───▶│   Artifact   │      │
│  │  (Weekly)    │    │   Pipeline   │    │    Store     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                             │                    │               │
│                             ▼                    ▼               │
│                      ┌──────────────┐    ┌──────────────┐      │
│                      │  ML Models   │    │   Analysis   │      │
│                      │  (Registry)  │    │   Results    │      │
│                      └──────────────┘    └──────────────┘      │
│                                                  │               │
│                                                  ▼               │
│                                          ┌──────────────┐      │
│                                          │ Web Exporter │      │
│                                          └──────────────┘      │
│                                                  │               │
│                                                  ▼               │
│                                          ┌──────────────┐      │
│                                          │ Static HTML  │      │
│                                          │   /JSON      │      │
│                                          └──────────────┘      │
│                                                  │               │
│                                                  ▼               │
│                                          ┌──────────────┐      │
│                                          │  Web Server  │      │
│                                          │  (nginx/S3)  │      │
│                                          └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Environment Setup

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Initialize ZenML
zenml init
```

### Configure API Keys

Create a `.env` file with your API keys:

```bash
# Blockchain Explorer APIs
ETHERSCAN_API_KEY=your_etherscan_key
BSCSCAN_API_KEY=your_bscscan_key
POLYGONSCAN_API_KEY=your_polygonscan_key

# Database (optional, for persistent storage)
DATABASE_URL=postgresql://user:pass@host:5432/stablecoin_db

# Auth0 (optional, for authenticated access)
AUTH0_DOMAIN=your_domain.auth0.com
AUTH0_CLIENT_ID=your_client_id
AUTH0_CLIENT_SECRET=your_client_secret
```

## Step 2: ZenML Configuration

### Configure ZenML Stack

```bash
# Register artifact store
zenml artifact-store register local_store \
    --flavor=local \
    --path=/path/to/artifacts

# Register orchestrator
zenml orchestrator register local_orchestrator \
    --flavor=local

# Create and activate stack
zenml stack register stablecoin_stack \
    --artifact-store=local_store \
    --orchestrator=local_orchestrator

zenml stack set stablecoin_stack
```

### For Production (Cloud Deployment)

```bash
# Example: AWS S3 artifact store
zenml artifact-store register s3_store \
    --flavor=s3 \
    --path=s3://your-bucket/artifacts

# Example: Kubernetes orchestrator
zenml orchestrator register k8s_orchestrator \
    --flavor=kubernetes \
    --kubernetes_context=your-context
```

## Step 3: Initial Pipeline Run

### Run the Master Pipeline

```bash
# Run with default parameters
python -c "from pipelines.master_pipeline import master_pipeline; master_pipeline()"

# Or with custom parameters
python -c "
from pipelines.master_pipeline import master_pipeline
master_pipeline(
    stablecoins=['USDC', 'USDT'],
    date_range_days=7,
    max_records=1000
)
"
```

### Verify Results

```bash
# List pipeline runs
zenml pipeline runs list

# Check artifacts
zenml artifact list
```

## Step 4: Configure Scheduled Execution

### Option A: System Cron

Add to crontab (`crontab -e`):

```bash
# Run master pipeline every Sunday at 2 AM
0 2 * * 0 /path/to/venv/bin/python /path/to/run_pipeline.py >> /var/log/stablecoin_pipeline.log 2>&1
```

Create `run_pipeline.py`:

```python
#!/usr/bin/env python
"""Scheduled pipeline runner."""
import os
import sys
from datetime import datetime

# Set up environment
os.chdir('/path/to/project')
sys.path.insert(0, '/path/to/project')

from pipelines.master_pipeline import master_pipeline
from notebooks.web_exporter import export_analysis_to_web

def main():
    print(f"Starting pipeline run at {datetime.now()}")
    
    # Run pipeline
    result = master_pipeline(
        stablecoins=['USDC', 'USDT'],
        date_range_days=7,
    )
    
    # Export to web
    export_analysis_to_web(
        run_id=result.id,
        analysis_results=result.artifacts,
        export_dir='/var/www/stablecoin-analysis/exports'
    )
    
    print(f"Pipeline completed at {datetime.now()}")

if __name__ == '__main__':
    main()
```

### Option B: ZenML Scheduling

```python
from zenml.pipelines import Schedule

# Create weekly schedule
schedule = Schedule(
    cron_expression="0 2 * * 0",  # Every Sunday at 2 AM
    name="weekly_stablecoin_analysis"
)

# Run pipeline with schedule
master_pipeline.with_options(schedule=schedule)()
```

## Step 5: Web Export Configuration

### Export Static Files

```python
from notebooks.web_exporter import WebExporter, ExportConfig

config = ExportConfig(
    export_dir=Path("/var/www/stablecoin-analysis/exports"),
    create_latest_symlink=True,
    include_visualizations=True,
    include_predictions=True,
)

exporter = WebExporter(config)
exporter.export(
    run_id="run-123",
    analysis_results=results,
    predictions=predictions,
    visualizations=charts,
)
```

### Directory Structure

```
/var/www/stablecoin-analysis/
├── exports/
│   ├── latest -> run-20241212-020000/
│   ├── run-20241212-020000/
│   │   ├── manifest.json
│   │   ├── analysis/
│   │   │   ├── activity_breakdown.json
│   │   │   ├── stablecoin_comparison.json
│   │   │   ├── holder_metrics.json
│   │   │   └── chain_metrics.json
│   │   ├── predictions/
│   │   │   ├── sov_predictions.json
│   │   │   └── wallet_classifications.json
│   │   ├── visualizations/
│   │   │   ├── activity_pie.html
│   │   │   ├── stablecoin_bar.html
│   │   │   └── ...
│   │   └── summary/
│   │       ├── dashboard.html
│   │       └── conclusions.json
│   └── run-20241205-020000/
│       └── ...
└── index.html
```

## Step 6: Web Server Configuration

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name stablecoin-analysis.example.com;
    
    root /var/www/stablecoin-analysis;
    index index.html;
    
    # Serve latest dashboard
    location / {
        try_files $uri $uri/ /exports/latest/summary/dashboard.html;
    }
    
    # API for JSON data
    location /api/ {
        alias /var/www/stablecoin-analysis/exports/latest/;
        add_header Content-Type application/json;
        add_header Access-Control-Allow-Origin *;
    }
    
    # Static visualizations
    location /charts/ {
        alias /var/www/stablecoin-analysis/exports/latest/visualizations/;
    }
    
    # Historical exports
    location /history/ {
        alias /var/www/stablecoin-analysis/exports/;
        autoindex on;
    }
}
```

### AWS S3 + CloudFront (Alternative)

```bash
# Sync exports to S3
aws s3 sync /var/www/stablecoin-analysis/exports/ s3://your-bucket/exports/ \
    --delete \
    --cache-control "max-age=3600"

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
    --distribution-id YOUR_DIST_ID \
    --paths "/exports/latest/*"
```

## Step 7: Monitoring and Alerts

### Model Performance Monitoring

Configure thresholds in `config/ml_thresholds.json`:

```json
{
    "precision_min": 0.70,
    "recall_min": 0.65,
    "f1_min": 0.67,
    "auc_min": 0.75,
    "relative_drop_max": 0.10
}
```

### Alert Configuration

```python
# In pipelines/model_monitor.py
from pipelines.model_monitor import ModelPerformanceMonitor

monitor = ModelPerformanceMonitor()
monitor.check_and_alert(
    model_name="sov_predictor",
    alert_channels=["email", "slack"]
)
```

### Health Check Endpoint

Add to your web server:

```python
# health_check.py
from datetime import datetime, timedelta
from pathlib import Path

def check_health():
    latest = Path("/var/www/stablecoin-analysis/exports/latest")
    manifest = latest / "manifest.json"
    
    if not manifest.exists():
        return {"status": "error", "message": "No exports found"}
    
    import json
    with open(manifest) as f:
        data = json.load(f)
    
    export_time = datetime.fromisoformat(data["export_timestamp"])
    age = datetime.now() - export_time
    
    if age > timedelta(days=8):
        return {"status": "warning", "message": f"Export is {age.days} days old"}
    
    return {"status": "healthy", "last_export": data["export_timestamp"]}
```

## Troubleshooting

### Common Issues

1. **Pipeline fails to collect data**
   - Check API keys are valid
   - Verify rate limits aren't exceeded
   - Check network connectivity

2. **ML model performance degraded**
   - Review recent data for distribution shifts
   - Check for data quality issues
   - Consider retraining with recent data

3. **Export fails**
   - Check disk space
   - Verify write permissions
   - Check for file size limits

### Logs

```bash
# ZenML logs
zenml pipeline runs logs <run_id>

# Application logs
tail -f /var/log/stablecoin_pipeline.log

# Web server logs
tail -f /var/log/nginx/access.log
```

## Security Considerations

1. **API Keys**: Store in environment variables or secrets manager
2. **Database**: Use encrypted connections, restrict access
3. **Web Access**: Consider authentication for sensitive data
4. **Wallet Addresses**: Always mask in logs and exports (first 6 + last 4 chars)

## Maintenance

### Weekly Tasks
- Review pipeline run logs
- Check model performance metrics
- Verify export completeness

### Monthly Tasks
- Review and archive old exports
- Update dependencies
- Check for API changes

### Quarterly Tasks
- Retrain ML models with fresh data
- Review and update thresholds
- Performance optimization review
