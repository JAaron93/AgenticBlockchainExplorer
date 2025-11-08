#!/usr/bin/env python3
"""
Production Configuration Validator

Run this script to validate your production configuration before deployment.
Usage: python config/validate_production.py [path/to/config.json]
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple


def validate_production_config(config_path: str = "./config.json") -> Tuple[bool, List[str]]:
    """
    Validate production configuration settings.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return False, [f"Configuration file not found: {config_path}"]
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    
    # Check app environment
    if config.get("app", {}).get("env") != "production":
        issues.append("❌ app.env must be 'production'")
    
    # Check debug mode
    if config.get("app", {}).get("debug") is not False:
        issues.append("❌ app.debug must be false in production")
    
    # Check secret key
    secret_key = config.get("app", {}).get("secret_key", "")
    if len(secret_key) < 32:
        issues.append("❌ app.secret_key must be at least 32 characters")
    if "example" in secret_key.lower() or "change" in secret_key.lower():
        issues.append("❌ app.secret_key appears to be a placeholder - generate a strong key")
    
    # Check cookie security
    if config.get("session", {}).get("cookie_secure") is not True:
        issues.append("❌ session.cookie_secure must be true (requires HTTPS)")
    
    # Check cookie_samesite
    samesite = config.get("session", {}).get("cookie_samesite")
    if samesite not in ["strict", "lax"]:
        issues.append("⚠️  session.cookie_samesite should be 'strict' or 'lax' for production")
    
    # Check CORS origins
    cors_origins = config.get("cors", {}).get("allowed_origins", [])
    for origin in cors_origins:
        if "localhost" in origin or "127.0.0.1" in origin:
            issues.append(f"❌ CORS origin contains localhost: {origin}")
        if not origin.startswith("https://"):
            issues.append(f"⚠️  CORS origin should use HTTPS: {origin}")
    
    # Check Auth0 URLs
    callback_url = config.get("auth0", {}).get("callback_url", "")
    logout_url = config.get("auth0", {}).get("logout_url", "")
    
    if "localhost" in callback_url:
        issues.append("❌ auth0.callback_url contains localhost")
    if "localhost" in logout_url:
        issues.append("❌ auth0.logout_url contains localhost")
    
    if not callback_url.startswith("https://"):
        issues.append("⚠️  auth0.callback_url should use HTTPS")
    if not logout_url.startswith("https://"):
        issues.append("⚠️  auth0.logout_url should use HTTPS")
    
    # Check database URL
    db_url = config.get("database", {}).get("url", "")
    if "<USER>" in db_url or "<PASSWORD>" in db_url or "<DB_HOST>" in db_url:
        issues.append("❌ database.url contains placeholders - replace with actual values")
    if "user:password" in db_url:
        issues.append("❌ database.url appears to use default credentials")
    
    # Check logging level
    log_level = config.get("logging", {}).get("level", "INFO")
    if log_level not in ["WARNING", "ERROR", "CRITICAL"]:
        issues.append(f"⚠️  logging.level is '{log_level}' - consider 'WARNING' or 'ERROR' for production")
    
    # Check API keys
    for explorer in config.get("explorers", []):
        api_key = explorer.get("api_key", "")
        if "YOUR_" in api_key or not api_key:
            issues.append(f"❌ {explorer.get('name', 'Unknown')} API key is a placeholder")
    
    return len(issues) == 0, issues


def main():
    """Main entry point."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./config.json"
    
    print(f"Validating production configuration: {config_path}")
    print("=" * 60)
    
    is_valid, issues = validate_production_config(config_path)
    
    if is_valid:
        print("✅ All production configuration checks passed!")
        print("\nYour configuration appears ready for production deployment.")
        print("Remember to:")
        print("  - Store secrets in environment variables")
        print("  - Never commit production config to version control")
        print("  - Test in a staging environment first")
        return 0
    else:
        print("⚠️  Production configuration issues found:\n")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n" + "=" * 60)
        print("Please review config/README.md for production deployment guidance.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
