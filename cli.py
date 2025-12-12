#!/usr/bin/env python3
"""
CLI script for running the blockchain stablecoin explorer agent standalone.

This script allows running the data collection agent without the web interface,
useful for testing, scheduled jobs, or command-line automation.

Usage:
    python cli.py --config config.json
    python cli.py --config config.json --output ./results
    python cli.py --config config.json --explorers etherscan,bscscan
    python cli.py --config config.json --stablecoins USDC
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Optional

from config.loader import ConfigurationManager
from config.models import Config
from core.logging import configure_logging, get_logger
from core.orchestrator import AgentOrchestrator, RunConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Blockchain Stablecoin Explorer - CLI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python cli.py

  # Run with custom config file
  python cli.py --config /path/to/config.json

  # Run with specific explorers only
  python cli.py --explorers etherscan,polygonscan

  # Run with specific stablecoins only
  python cli.py --stablecoins USDC

  # Limit records per explorer
  python cli.py --max-records 100

  # Custom output directory
  python cli.py --output ./my-results

  # Verbose output
  python cli.py --verbose
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="./config.json",
        help="Path to configuration file (default: ./config.json)"
    )

    parser.add_argument(
        "--env-file", "-e",
        type=str,
        default="./.env",
        help="Path to .env file (default: ./.env)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for JSON results (overrides config)"
    )

    parser.add_argument(
        "--explorers",
        type=str,
        help="Comma-separated list of explorers to use (e.g., etherscan,bscscan)"
    )

    parser.add_argument(
        "--stablecoins",
        type=str,
        help="Comma-separated list of stablecoins to collect (e.g., USDC,USDT)"
    )

    parser.add_argument(
        "--max-records",
        type=int,
        help="Maximum records per explorer (overrides config)"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Custom run ID (default: auto-generated UUID)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )

    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results as JSON to stdout"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running collection"
    )

    return parser.parse_args()


def load_configuration(
    config_path: str,
    env_file: str,
    output_override: Optional[str] = None
) -> Config:
    """Load and validate configuration.
    
    Args:
        config_path: Path to JSON config file.
        env_file: Path to .env file.
        output_override: Optional output directory override.
        
    Returns:
        Validated Config object.
    """
    config_manager = ConfigurationManager(
        config_path=config_path,
        env_file=env_file
    )
    
    config = config_manager.load_config()
    config_manager.validate_config(config)
    
    # Apply output directory override
    if output_override:
        config.output.directory = output_override
        # Ensure directory exists
        Path(output_override).mkdir(parents=True, exist_ok=True)
    
    return config


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Build run configuration from CLI arguments.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        RunConfig with any overrides from CLI.
    """
    run_config = RunConfig()
    
    if args.explorers:
        run_config.explorers = [
            e.strip() for e in args.explorers.split(",") if e.strip()
        ]
    
    if args.stablecoins:
        run_config.stablecoins = [
            s.strip().upper() for s in args.stablecoins.split(",") if s.strip()
        ]
    
    if args.max_records:
        run_config.max_records_per_explorer = args.max_records
    
    return run_config


async def run_agent(
    config: Config,
    run_config: RunConfig,
    run_id: str,
    logger
) -> dict:
    """Run the agent and return results.
    
    Args:
        config: Application configuration.
        run_config: Run-specific configuration.
        run_id: Unique run identifier.
        logger: Logger instance.
        
    Returns:
        Dictionary with run results.
    """
    logger.info(f"Starting agent run: {run_id}")
    logger.info(f"Explorers: {[e.name for e in config.explorers]}")
    logger.info(f"Stablecoins: {list(config.stablecoins.keys())}")
    
    # Create orchestrator without database (standalone mode)
    orchestrator = AgentOrchestrator(
        config=config,
        run_id=run_id,
        db_manager=None,  # No database in CLI mode
        user_id="cli",
        run_config=run_config,
    )
    
    # Run collection
    report = await orchestrator.run()
    
    return report.to_dict()


def print_results(results: dict, json_output: bool = False) -> None:
    """Print results to console.
    
    Args:
        results: Results dictionary from agent run.
        json_output: If True, output as JSON.
    """
    if json_output:
        print(json.dumps(results, indent=2, default=str))
        return
    
    print("\n" + "=" * 60)
    print("COLLECTION RESULTS")
    print("=" * 60)
    print(f"Run ID: {results['run_id']}")
    print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
    print(f"Duration: {results['duration_seconds']:.2f} seconds")
    print(f"Total Records: {results['total_records']}")
    
    if results.get('output_file_path'):
        print(f"Output File: {results['output_file_path']}")
    
    print("\nRecords by Source:")
    for source, count in results.get('records_by_source', {}).items():
        print(f"  - {source}: {count}")
    
    if results.get('explorers_failed'):
        print("\nFailed Explorers:")
        for explorer in results['explorers_failed']:
            print(f"  - {explorer}")
    
    if results.get('errors'):
        print("\nErrors:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more errors")
    
    print("=" * 60 + "\n")


def main() -> int:
    """Main entry point for CLI.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args()
    
    # Determine log level
    if args.quiet:
        log_level = "ERROR"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    # Configure logging
    configure_logging(
        level=log_level,
        fmt="text",  # Use text format for CLI
        service_name="stablecoin-explorer-cli"
    )
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_configuration(
            config_path=args.config,
            env_file=args.env_file,
            output_override=args.output
        )
        
        # Build run configuration
        run_config = build_run_config(args)
        
        # Generate run ID
        run_id = args.run_id or str(uuid.uuid4())
        
        # Dry run - just validate and exit
        if args.dry_run:
            print("Configuration validated successfully!")
            print(f"Explorers: {[e.name for e in config.explorers]}")
            print(f"Stablecoins: {list(config.stablecoins.keys())}")
            print(f"Output directory: {config.output.directory}")
            if run_config.explorers:
                print(f"Filtered explorers: {run_config.explorers}")
            if run_config.stablecoins:
                print(f"Filtered stablecoins: {run_config.stablecoins}")
            return 0
        
        # Run the agent
        logger.info("Starting data collection...")
        
        results = asyncio.run(run_agent(
            config=config,
            run_config=run_config,
            run_id=run_id,
            logger=logger
        ))
        
        # Print results
        if not args.quiet:
            print_results(results, json_output=args.json_output)
        elif args.json_output:
            print(json.dumps(results, indent=2, default=str))
        
        # Return appropriate exit code
        if results.get('success'):
            logger.info("Agent run completed successfully")
            return 0
        else:
            logger.error("Agent run completed with failures")
            return 1
            
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("Agent run interrupted by user")
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
