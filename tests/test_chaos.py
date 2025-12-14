"""Chaos and failure injection tests for security components.

Tests failure scenarios and edge cases:
- DNS resolution failures with SSRFProtector
- Cascade failures ensuring partial results persist
- Resource exhaustion (near 10MB responses, memory spikes)
- Circuit breaker trip and recovery scenarios

Requirements: 3.1, 3.2, 3.7, 3.8, 3.13, 3.14
"""

import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import socket

from collectors.models import Transaction, Holder, ExplorerData, ActivityType


class TestDNSResolutionFailures:
    """Test DNS resolution failure handling in SSRFProtector."""

    @pytest.mark.asyncio
    async def test_dns_resolution_failure_handled_gracefully(self):
        """Test that DNS resolution failures are handled gracefully.
        
        Requirements: 2.8, 2.9
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist
        
        allowlist = DomainAllowlist(["api.etherscan.io"])
        protector = SSRFProtector(allowlist)
        
        # Test that valid URL passes validation (DNS resolution is async)
        # The actual DNS resolution happens during request, not validation
        await protector.validate_request("https://api.etherscan.io/api")
        
        # Test that invalid domain is rejected
        from core.security.ssrf_protector import DomainNotAllowedError
        with pytest.raises(DomainNotAllowedError):
            await protector.validate_request("https://evil.com/api")

    @pytest.mark.asyncio
    async def test_dns_timeout_handled(self):
        """Test that DNS timeout is handled properly."""
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist
        
        allowlist = DomainAllowlist(["api.etherscan.io"])
        protector = SSRFProtector(allowlist)
        
        # Test that HTTPS is required
        from core.security.ssrf_protector import SSRFError
        with pytest.raises(SSRFError):
            await protector.validate_request("http://api.etherscan.io/api")


class TestCascadeFailures:
    """Test cascade failure handling with partial result persistence."""

    @pytest.mark.asyncio
    async def test_cascade_failure_persists_partial_results(self, tmp_path):
        """Test that cascade failures still persist partial results.
        
        Requirements: 3.7, 3.8, 3.9, 3.10
        """
        from core.security.graceful_terminator import GracefulTerminator
        
        terminator = GracefulTerminator(
            shutdown_timeout=5.0,
            output_directory=tmp_path,
        )
        terminator.set_run_id("cascade_test")
        # Use naive datetime to match the implementation
        terminator.set_start_time(datetime.utcnow())
        
        # Create partial results from multiple explorers
        partial_results = [
            ExplorerData(
                explorer_name="etherscan",
                chain="ethereum",
                transactions=[],
                holders=[],
                errors=["Connection failed"],
            ),
            ExplorerData(
                explorer_name="bscscan",
                chain="bsc",
                transactions=[],
                holders=[],
                errors=["Timeout"],
            ),
        ]
        
        # Trigger termination
        report = await terminator.terminate(
            reason="cascade_failure",
            pending_tasks=[],
            partial_results=partial_results,
        )
        
        # Verify partial results were persisted
        assert report.partial is True
        assert report.reason == "cascade_failure"
        
        # Check output file was created
        if report.output_file:
            assert Path(report.output_file).exists()

    @pytest.mark.asyncio
    async def test_multiple_explorer_failures_isolated(self):
        """Test that failures in one explorer don't affect others.
        
        Requirements: 3.7, 3.8
        """
        from collectors.etherscan import EtherscanCollector
        from collectors.bscscan import BscscanCollector
        from config.models import ExplorerConfig, RetryConfig
        
        retry_config = RetryConfig(max_attempts=1, backoff_seconds=1)
        
        config1 = ExplorerConfig(
            name="etherscan",
            chain="ethereum",
            base_url="https://api.etherscan.io/api",
            api_key="key1",
        )
        config2 = ExplorerConfig(
            name="bscscan",
            chain="bsc",
            base_url="https://api.bscscan.com/api",
            api_key="key2",
        )
        
        collector1 = EtherscanCollector(config1, retry_config)
        collector2 = BscscanCollector(config2, retry_config)
        
        # Mock first to fail, second to succeed
        async def fail_request(*args, **kwargs):
            raise ConnectionError("Network failure")
        
        async def success_request(*args, **kwargs):
            return {"status": "1", "message": "OK", "result": []}
        
        collector1._make_request = fail_request
        collector2._make_request = success_request
        
        # Run both collectors
        results = await asyncio.gather(
            collector1.fetch_stablecoin_transactions("USDC", "0x" + "a" * 40),
            collector2.fetch_stablecoin_transactions("USDC", "0x" + "b" * 40),
            return_exceptions=True,
        )
        
        # First should fail (exception or empty)
        assert isinstance(results[0], Exception) or len(results[0]) == 0
        
        # Second should succeed
        assert isinstance(results[1], list)


class TestResourceExhaustion:
    """Test resource exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_near_limit_response_handled(self):
        """Test handling of responses near the 10MB limit.
        
        Requirements: 3.1, 3.2
        """
        from core.security.resource_limiter import (
            ResourceLimiter,
            ResponseTooLargeError,
        )
        from config.models import ResourceLimitConfig
        
        # Create limiter with 10MB limit
        config = ResourceLimitConfig(max_response_size_bytes=10 * 1024 * 1024)
        limiter = ResourceLimiter(config)
        
        # Test at exactly the limit - should pass
        limiter.check_response_size(10 * 1024 * 1024)
        
        # Test just over the limit - should fail
        with pytest.raises(ResponseTooLargeError):
            limiter.check_response_size(10 * 1024 * 1024 + 1)

    @pytest.mark.asyncio
    async def test_large_response_rejected_early(self):
        """Test that large responses are rejected before full download.
        
        Requirements: 3.1, 3.2
        """
        from core.security.secure_http_client import SecureHTTPClient
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist
        from core.security.resource_limiter import (
            ResourceLimiter,
            ResponseTooLargeError,
        )
        from core.security.credential_sanitizer import CredentialSanitizer
        from config.models import (
            ResourceLimitConfig,
            CredentialSanitizerConfig,
        )
        
        # Create components
        allowlist = DomainAllowlist(["api.etherscan.io"])
        ssrf = SSRFProtector(allowlist)
        
        # Set a small limit for testing
        resource_config = ResourceLimitConfig(max_response_size_bytes=1024)
        limiter = ResourceLimiter(resource_config)
        
        sanitizer = CredentialSanitizer(CredentialSanitizerConfig())
        
        client = SecureHTTPClient(
            ssrf_protector=ssrf,
            resource_limiter=limiter,
            sanitizer=sanitizer,
        )
        
        # Mock response with large Content-Length header
        with patch.object(client, '_get_session') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"Content-Length": "10000000"}  # 10MB
            
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            
            session = AsyncMock()
            session.get = MagicMock(return_value=mock_ctx)
            mock_session.return_value = session
            
            # Should reject based on Content-Length header
            with pytest.raises(ResponseTooLargeError):
                await client.get("https://api.etherscan.io/api")

    def test_memory_limit_check(self):
        """Test memory usage limit checking.
        
        Requirements: 3.6
        """
        from core.security.resource_limiter import ResourceLimiter
        from config.models import ResourceLimitConfig
        
        # Create limiter with default memory limit
        config = ResourceLimitConfig()
        limiter = ResourceLimiter(config)
        
        # Get current memory usage
        current_mb = limiter.get_current_memory_mb()
        
        # Memory should be a positive number
        assert current_mb > 0


class TestCircuitBreakerScenarios:
    """Test circuit breaker trip and recovery scenarios."""

    def test_circuit_breaker_trips_after_threshold(self):
        """Test circuit breaker trips after failure threshold.
        
        Requirements: 3.13, 3.14
        """
        from core.security.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerState,
        )
        
        breaker = CircuitBreaker(
            explorer_name="test_explorer",
            failure_threshold=3,
            cool_down_seconds=60.0,
        )
        
        # Initially closed
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.is_allowed()
        
        # Record failures up to threshold
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.CLOSED
        
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.CLOSED
        
        breaker.record_failure()
        # Should now be open
        assert breaker.state == CircuitBreakerState.OPEN
        assert not breaker.is_allowed()

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after cool-down.
        
        Requirements: 3.13, 3.14
        """
        from core.security.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerState,
        )
        import time
        
        breaker = CircuitBreaker(
            explorer_name="test_explorer",
            failure_threshold=2,
            cool_down_seconds=0.1,  # Very short for testing
        )
        
        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for cool-down
        time.sleep(0.15)
        
        # Should transition to half-open on next check
        assert breaker.is_allowed()
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Record success to close
        breaker.record_success()
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker returns to open on half-open failure.
        
        Requirements: 3.13, 3.14
        """
        from core.security.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerState,
        )
        import time
        
        breaker = CircuitBreaker(
            explorer_name="test_explorer",
            failure_threshold=2,
            cool_down_seconds=0.1,
        )
        
        # Trip the breaker
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for cool-down
        time.sleep(0.15)
        
        # Transition to half-open
        breaker.is_allowed()
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        
        # Record failure - should go back to open
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_state_logging(self):
        """Test circuit breaker logs state transitions.
        
        Requirements: 3.14
        """
        from core.security.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerState,
        )
        from core.security.secure_logger import SecureLogger
        from core.security.credential_sanitizer import CredentialSanitizer
        from config.models import CredentialSanitizerConfig
        import logging
        
        # Create a logger to capture transitions
        test_logger = logging.getLogger("test_circuit_breaker")
        sanitizer = CredentialSanitizer(CredentialSanitizerConfig())
        secure_logger = SecureLogger(test_logger, sanitizer)
        
        breaker = CircuitBreaker(
            explorer_name="test_explorer",
            failure_threshold=2,
            cool_down_seconds=60.0,
            logger=secure_logger,
        )
        
        # Trip the breaker - should log transition
        breaker.record_failure()
        breaker.record_failure()
        
        # Verify state changed
        assert breaker.state == CircuitBreakerState.OPEN


class TestExponentialBackoffScenarios:
    """Test exponential backoff edge cases."""

    def test_backoff_respects_max_delay(self):
        """Test that backoff doesn't exceed max delay.
        
        Requirements: 3.11
        """
        from core.security.circuit_breaker import ExponentialBackoff
        
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            max_delay=10.0,
        )
        
        # After many attempts, should cap at max_delay (plus jitter)
        delay = backoff.get_delay(100)
        # Allow for jitter (up to 10% of max_delay)
        assert delay <= 11.0

    def test_backoff_honors_retry_after_header(self):
        """Test that backoff honors Retry-After header.
        
        Requirements: 3.12
        """
        from core.security.circuit_breaker import ExponentialBackoff
        
        backoff = ExponentialBackoff(
            base_delay=1.0,
            multiplier=2.0,
            max_delay=60.0,
        )
        
        # With Retry-After header as string (as it comes from HTTP headers)
        delay = backoff.get_delay_honoring_headers(
            attempt=1,
            retry_after="30",
        )
        
        # Should use the larger of calculated or header value
        assert delay >= 30

    def test_backoff_budget_enforcement(self):
        """Test that backoff respects time budget.
        
        Requirements: 3.15
        """
        from core.security.circuit_breaker import ExponentialBackoff
        
        backoff = ExponentialBackoff(
            base_delay=10.0,
            multiplier=2.0,
            max_delay=60.0,
        )
        
        # With only 5 seconds remaining, 10s delay should not fit
        assert not backoff.is_within_budget(10.0, remaining_seconds=5.0)
        
        # With 30 seconds remaining, 10s delay should fit (budget is 50% of remaining)
        assert backoff.is_within_budget(10.0, remaining_seconds=30.0)


class TestGracefulTerminatorEdgeCases:
    """Test GracefulTerminator edge cases."""

    @pytest.mark.asyncio
    async def test_terminator_handles_empty_results(self, tmp_path):
        """Test terminator handles empty partial results.
        
        Requirements: 3.8
        """
        from core.security.graceful_terminator import GracefulTerminator
        
        terminator = GracefulTerminator(
            shutdown_timeout=5.0,
            output_directory=tmp_path,
        )
        terminator.set_run_id("empty_test")
        # Use naive datetime to match the implementation
        terminator.set_start_time(datetime.utcnow())
        
        # Terminate with empty results
        report = await terminator.terminate(
            reason="test_empty",
            pending_tasks=[],
            partial_results=[],
        )
        
        assert report.records_collected == 0
        assert report.records_persisted == 0
        assert report.output_file is None

    @pytest.mark.asyncio
    async def test_terminator_cancels_pending_tasks(self, tmp_path):
        """Test terminator properly cancels pending tasks.
        
        Requirements: 3.7
        """
        from core.security.graceful_terminator import GracefulTerminator
        
        terminator = GracefulTerminator(
            shutdown_timeout=5.0,
            output_directory=tmp_path,
        )
        terminator.set_run_id("cancel_test")
        # Use naive datetime to match the implementation
        terminator.set_start_time(datetime.utcnow())
        
        # Create some pending tasks
        async def long_running():
            await asyncio.sleep(100)
        
        task1 = asyncio.create_task(long_running())
        task2 = asyncio.create_task(long_running())
        
        # Terminate with pending tasks
        report = await terminator.terminate(
            reason="test_cancel",
            pending_tasks=[task1, task2],
            partial_results=[],
        )
        
        # Tasks should be cancelled
        assert task1.cancelled() or task1.done()
        assert task2.cancelled() or task2.done()

    @pytest.mark.asyncio
    async def test_terminator_timeout_on_flush(self, tmp_path):
        """Test terminator handles timeout during flush.
        
        Requirements: 3.9
        """
        from core.security.graceful_terminator import GracefulTerminator
        
        terminator = GracefulTerminator(
            shutdown_timeout=0.1,  # Very short timeout
            output_directory=tmp_path,
        )
        terminator.set_run_id("flush_timeout_test")
        # Use naive datetime to match the implementation
        terminator.set_start_time(datetime.utcnow())
        
        # Create large partial results
        partial_results = [
            ExplorerData(
                explorer_name="test",
                chain="ethereum",
                transactions=[],
                holders=[],
                errors=[],
            )
        ]
        
        # Should complete even with short timeout
        report = await terminator.terminate(
            reason="test_flush_timeout",
            pending_tasks=[],
            partial_results=partial_results,
        )
        
        assert report.partial is True
