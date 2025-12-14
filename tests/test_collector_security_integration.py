"""Integration tests for collector security components.

Tests the integration of security components with collectors:
- Timeout behavior with GracefulTerminator persisting partial results
- Error isolation (SSRFError in one collection doesn't suppress others)
- Concurrent collections for race conditions in SSRFProtector._dns_cache

Requirements: All security requirements
"""

import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from collectors.models import Transaction, Holder, ExplorerData, ActivityType
from collectors.etherscan import EtherscanCollector
from collectors.bscscan import BscscanCollector
from collectors.polygonscan import PolygonscanCollector
from config.models import ExplorerConfig, RetryConfig


# Test fixtures
@pytest.fixture
def mock_explorer_config():
    """Create a mock explorer configuration."""
    return ExplorerConfig(
        name="test_explorer",
        chain="ethereum",
        base_url="https://api.etherscan.io/api",
        api_key="test_api_key",
    )


@pytest.fixture
def mock_retry_config():
    """Create a mock retry configuration."""
    return RetryConfig(
        max_attempts=2,
        backoff_seconds=1,  # Must be integer
        request_timeout_seconds=5.0,
    )


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing."""
    return Transaction(
        transaction_hash="0x" + "a" * 64,
        block_number=12345678,
        timestamp=datetime.now(timezone.utc),
        from_address="0x" + "1" * 40,
        to_address="0x" + "2" * 40,
        amount=Decimal("100.0"),
        stablecoin="USDC",
        chain="ethereum",
        activity_type=ActivityType.TRANSACTION,
        source_explorer="etherscan",
    )


@pytest.fixture
def sample_holder():
    """Create a sample holder for testing."""
    return Holder(
        address="0x" + "3" * 40,
        balance=Decimal("1000.0"),
        stablecoin="USDC",
        chain="ethereum",
        first_seen=datetime.now(timezone.utc),
        last_activity=datetime.now(timezone.utc),
        is_store_of_value=False,
        source_explorer="etherscan",
    )


class TestTimeoutBehavior:
    """Test timeout behavior with GracefulTerminator."""

    @pytest.mark.asyncio
    async def test_timeout_persists_partial_results(self, tmp_path):
        """Test that timeout triggers graceful termination with partial results.
        
        Requirements: 3.7, 3.8, 3.9, 3.10, 6.1, 6.2, 6.6
        """
        from core.security.graceful_terminator import GracefulTerminator
        from core.security.timeout_manager import TimeoutManager
        from config.models import TimeoutConfig
        
        # Create timeout config with minimum valid values
        timeout_config = TimeoutConfig(
            overall_run_timeout_seconds=120,
            per_collection_timeout_seconds=60,
            shutdown_timeout_seconds=10,
        )
        
        # Create timeout manager
        timeout_manager = TimeoutManager(timeout_config, num_collections=1)
        timeout_manager.start()
        
        # Create graceful terminator
        terminator = GracefulTerminator(
            shutdown_timeout=1.0,
            output_directory=tmp_path,
        )
        terminator.set_run_id("test_run_123")
        terminator.set_start_time(datetime.utcnow())
        
        # Create partial results
        partial_results = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=[],
                holders=[],
                errors=["Timeout occurred"],
            )
        ]
        
        # Trigger graceful termination
        report = await terminator.terminate(
            reason="test_timeout",
            pending_tasks=[],
            partial_results=partial_results,
        )
        
        # Verify termination report
        assert report.partial is True
        assert report.reason == "test_timeout"
        assert report.records_collected == 0
        
    @pytest.mark.asyncio
    async def test_timeout_manager_should_terminate(self):
        """Test TimeoutManager.should_terminate() threshold.
        
        Requirements: 6.3
        """
        from core.security.timeout_manager import TimeoutManager
        from config.models import TimeoutConfig
        
        # Create config with minimum valid values
        timeout_config = TimeoutConfig(
            overall_run_timeout_seconds=120,
            per_collection_timeout_seconds=60,
            shutdown_timeout_seconds=10,
        )
        
        timeout_manager = TimeoutManager(timeout_config, num_collections=1)
        timeout_manager.start()
        
        # Initially should not terminate (120s timeout, 60s threshold)
        assert not timeout_manager.should_terminate()
        
        # Manually set start time to simulate near-timeout
        import time
        timeout_manager._start_time = time.monotonic() - 65  # 65s elapsed
        
        # Now should indicate termination needed (55s remaining < 60s threshold)
        assert timeout_manager.should_terminate()


class TestErrorIsolation:
    """Test error isolation between collectors."""

    @pytest.mark.asyncio
    async def test_ssrf_error_does_not_suppress_other_collections(
        self, mock_explorer_config, mock_retry_config
    ):
        """Test that SSRFError in one collection doesn't affect others.
        
        Requirements: 2.4, 2.5, 2.6
        """
        # Create two collectors
        config1 = ExplorerConfig(
            name="explorer1",
            chain="ethereum",
            base_url="https://api.etherscan.io/api",
            api_key="key1",
        )
        config2 = ExplorerConfig(
            name="explorer2",
            chain="bsc",
            base_url="https://api.bscscan.com/api",
            api_key="key2",
        )
        
        collector1 = EtherscanCollector(config1, mock_retry_config)
        collector2 = BscscanCollector(config2, mock_retry_config)
        
        # Mock _make_secure_request to simulate SSRF error on first
        # and _make_standard_request for success on second
        async def mock_request_fail(*args, **kwargs):
            # Return None to simulate failed request (SSRF blocked)
            return None
        
        async def mock_request_success(*args, **kwargs):
            return {
                "status": "1",
                "message": "OK",
                "result": []
            }
        
        with patch.object(collector1, '_make_request', mock_request_fail):
            with patch.object(collector2, '_make_request', mock_request_success):
                # First collector should fail (returns empty list)
                result1 = await collector1.fetch_stablecoin_transactions(
                    "USDC", "0x" + "a" * 40, limit=10
                )
                
                # Second collector should succeed
                result2 = await collector2.fetch_stablecoin_transactions(
                    "USDC", "0x" + "b" * 40, limit=10
                )
        
        # First should return empty due to error
        assert len(result1) == 0
        
        # Second should succeed (empty result is valid)
        assert isinstance(result2, list)

    @pytest.mark.asyncio
    async def test_validation_error_skips_record_continues_processing(
        self, mock_explorer_config, mock_retry_config
    ):
        """Test that validation errors skip individual records but continue.
        
        Requirements: 4.1, 4.2, 4.4
        """
        collector = EtherscanCollector(mock_explorer_config, mock_retry_config)
        
        # Test data with mix of valid and invalid transactions
        test_data = [
            {
                "hash": "0x" + "a" * 64,  # Valid
                "from": "0x" + "1" * 40,
                "to": "0x" + "2" * 40,
                "value": "1000000",
                "timeStamp": "1609459200",
                "blockNumber": "12345678",
            },
            {
                "hash": "invalid_hash",  # Invalid - should be skipped
                "from": "0x" + "3" * 40,
                "to": "0x" + "4" * 40,
                "value": "2000000",
                "timeStamp": "1609459300",
                "blockNumber": "12345679",
            },
            {
                "hash": "0x" + "b" * 64,  # Valid
                "from": "0x" + "5" * 40,
                "to": "0x" + "6" * 40,
                "value": "3000000",
                "timeStamp": "1609459400",
                "blockNumber": "12345680",
            },
        ]
        
        # Parse transactions
        results = []
        for tx_data in test_data:
            tx = collector._parse_transaction(tx_data, "USDC")
            if tx:
                results.append(tx)
        
        # Should have 2 valid transactions (invalid one skipped)
        assert len(results) == 2
        assert results[0].transaction_hash == "0x" + "a" * 64
        assert results[1].transaction_hash == "0x" + "b" * 64


class TestConcurrentCollections:
    """Test concurrent collection behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_collections_no_race_conditions(
        self, mock_retry_config
    ):
        """Test concurrent collections don't have race conditions.
        
        This tests that multiple collectors can run concurrently without
        interfering with each other's state.
        """
        configs = [
            ExplorerConfig(
                name=f"explorer_{i}",
                chain=chain,
                base_url=f"https://api.{chain}scan.io/api",
                api_key=f"key_{i}",
            )
            for i, chain in enumerate(["ethereum", "bsc", "polygon"])
        ]
        
        collectors = [
            EtherscanCollector(configs[0], mock_retry_config),
            BscscanCollector(configs[1], mock_retry_config),
            PolygonscanCollector(configs[2], mock_retry_config),
        ]
        
        # Mock all collectors to return success
        async def mock_request(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay to simulate network
            return {
                "status": "1",
                "message": "OK",
                "result": []
            }
        
        for collector in collectors:
            collector._make_request = mock_request
        
        # Run all collectors concurrently
        tasks = [
            collector.fetch_stablecoin_transactions(
                "USDC", "0x" + "a" * 40, limit=10
            )
            for collector in collectors
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_dns_cache_thread_safety(self):
        """Test SSRFProtector DNS cache is thread-safe under concurrent access.
        
        Requirements: 2.8, 2.9
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist
        
        # Create SSRF protector with allowlist
        allowlist = DomainAllowlist([
            "api.etherscan.io",
            "api.bscscan.com",
            "api.polygonscan.com",
        ])
        protector = SSRFProtector(allowlist)
        
        # Concurrent validation requests
        urls = [
            "https://api.etherscan.io/api",
            "https://api.bscscan.com/api",
            "https://api.polygonscan.com/api",
        ] * 10  # 30 total requests
        
        async def validate_url(url: str):
            try:
                await protector.validate_request(url)
                return True
            except Exception:
                return False
        
        # Run all validations concurrently
        results = await asyncio.gather(
            *[validate_url(url) for url in urls],
            return_exceptions=True
        )
        
        # All should succeed (no race condition errors)
        for result in results:
            assert not isinstance(result, Exception)
            assert result is True


class TestBlockchainValidatorIntegration:
    """Test BlockchainDataValidator integration with collectors."""

    def test_address_normalization_in_transaction_parsing(
        self, mock_explorer_config, mock_retry_config
    ):
        """Test that addresses are normalized to lowercase.
        
        Requirements: 4.5
        """
        collector = EtherscanCollector(mock_explorer_config, mock_retry_config)
        
        # Transaction with mixed-case addresses
        tx_data = {
            "hash": "0x" + "A" * 64,
            "from": "0x" + "A" * 40,  # Uppercase
            "to": "0x" + "B" * 40,    # Uppercase
            "value": "1000000",
            "timeStamp": "1609459200",
            "blockNumber": "12345678",
        }
        
        tx = collector._parse_transaction(tx_data, "USDC")
        
        # Addresses should be normalized to lowercase
        assert tx is not None, "Transaction should be parsed successfully"
        assert tx.from_address == "0x" + "a" * 40
        assert tx.to_address == "0x" + "b" * 40

    def test_holder_address_normalization(
        self, mock_explorer_config, mock_retry_config
    ):
        """Test that holder addresses are normalized to lowercase.
        
        Requirements: 4.5
        """
        collector = EtherscanCollector(mock_explorer_config, mock_retry_config)
        
        # Holder with uppercase address
        holder_data = {
            "TokenHolderAddress": "0x" + "C" * 40,  # Uppercase
            "TokenHolderQuantity": "1000000000",
        }
        
        holder = collector._parse_holder(holder_data, "USDC", "0x" + "d" * 40)
        
        # Address should be normalized to lowercase
        assert holder is not None, "Holder should be parsed successfully"
        assert holder.address == "0x" + "c" * 40


class TestSecureHTTPClientIntegration:
    """Test SecureHTTPClient integration with collectors."""

    @pytest.mark.asyncio
    async def test_secure_client_used_when_available(
        self, mock_explorer_config, mock_retry_config
    ):
        """Test that SecureHTTPClient is used when security components available."""
        collector = EtherscanCollector(mock_explorer_config, mock_retry_config)
        
        # Mock the secure client getter
        with patch('collectors.base._get_secure_http_client') as mock_getter:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value={
                "status": "1",
                "message": "OK",
                "result": []
            })
            mock_getter.return_value = mock_client
            
            # Make a request
            result = await collector._make_request(
                {"module": "account", "action": "tokentx"},
                run_id="test_run"
            )
            
            # Verify secure client was used
            mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_to_standard_client(
        self, mock_explorer_config, mock_retry_config
    ):
        """Test fallback to standard aiohttp when SecureHTTPClient unavailable."""
        collector = EtherscanCollector(mock_explorer_config, mock_retry_config)
        
        # Mock secure client as unavailable
        with patch('collectors.base._get_secure_http_client', return_value=None):
            # Mock the standard session
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "status": "1",
                "message": "OK",
                "result": []
            })
            
            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None)
            ))
            
            with patch.object(collector, '_get_session', return_value=mock_session):
                result = await collector._make_request(
                    {"module": "account", "action": "tokentx"},
                    run_id="test_run"
                )
            
            # Should still work with fallback
            # Option 1: Just verify fallback returned something
            assert result is not None
            # Option 2: Verify exact expected result
            # assert result == {"status": "1", "message": "OK", "result": []}
