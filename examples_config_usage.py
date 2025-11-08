#!/usr/bin/env python3
"""Example script demonstrating configuration usage."""

from config import ConfigurationManager

def main():
    """Demonstrate configuration loading and usage."""
    print("=" * 60)
    print("Configuration Management Example")
    print("=" * 60)
    
    try:
        # Initialize configuration manager
        print("\n1. Loading configuration from config.json...")
        manager = ConfigurationManager(config_path="./config.json")
        config = manager.load_config()
        print("✓ Configuration loaded successfully!")
        
        # Display application settings
        print("\n2. Application Settings:")
        print(f"   Environment: {config.app.env}")
        print(f"   Host: {config.app.host}")
        print(f"   Port: {config.app.port}")
        print(f"   Debug: {config.app.debug}")
        
        # Display database settings
        print("\n3. Database Settings:")
        print(f"   URL: {config.database.url}")
        print(f"   Pool Size: {config.database.pool_size}")
        print(f"   Max Overflow: {config.database.max_overflow}")
        
        # Display Auth0 settings
        print("\n4. Auth0 Settings:")
        print(f"   Domain: {config.auth0.domain}")
        print(f"   Client ID: {config.auth0.client_id}")
        print(f"   Audience: {config.auth0.audience}")
        
        # Display explorer configurations
        print("\n5. Blockchain Explorers:")
        for explorer in config.explorers:
            print(f"   - {explorer.name.upper()}")
            print(f"     Chain: {explorer.chain}")
            print(f"     URL: {explorer.base_url}")
            print(f"     Type: {explorer.type}")
        
        # Display stablecoin addresses
        print("\n6. Stablecoin Contract Addresses:")
        for coin_name, coin_config in config.stablecoins.items():
            print(f"   {coin_name}:")
            print(f"     Ethereum: {coin_config.ethereum}")
            print(f"     BSC: {coin_config.bsc}")
            print(f"     Polygon: {coin_config.polygon}")
        
        # Display output settings
        print("\n7. Output Settings:")
        print(f"   Directory: {config.output.directory}")
        print(f"   Max Records/Explorer: "
              f"{config.output.max_records_per_explorer}")
        
        # Display retry settings
        print("\n8. Retry Settings:")
        print(f"   Max Attempts: {config.retry.max_attempts}")
        print(f"   Backoff: {config.retry.backoff_seconds}s")
        print(f"   Timeout: {config.retry.request_timeout_seconds}s")
        print(f"   Max Concurrent: {config.retry.max_concurrent_requests}")
        
        # Demonstrate helper methods
        print("\n9. Using Helper Methods:")
        
        # Get specific explorer
        etherscan = manager.get_explorer_by_name("etherscan")
        if etherscan:
            print(f"   Etherscan found: {etherscan.base_url}")
        
        # Get explorer by chain
        bsc_explorer = manager.get_explorer_by_chain("bsc")
        if bsc_explorer:
            print(f"   BSC explorer: {bsc_explorer.name}")
        
        # Validate configuration
        print("\n10. Validating Configuration:")
        is_valid = manager.validate_config(config)
        print(f"    Configuration is valid: {is_valid}")
        
        print("\n" + "=" * 60)
        print("Configuration example completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Configuration file not found")
        print(f"  {e}")
        print("\n  Please copy config.example.json to config.json:")
        print("  $ cp config.example.json config.json")
        
    except Exception as e:
        print(f"\n✗ Error loading configuration:")
        print(f"  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
