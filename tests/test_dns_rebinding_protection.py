"""
Unit tests for DNS rebinding protection in SSRF protector.

These tests verify the DNS resolution pinning, rebinding detection,
and redirect validation functionality.

**Requirements: 2.8, 2.9**
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from core.security.ssrf_protector import (
    DNSRebindingError,
    DNSResolutionError,
    DomainAllowlist,
    DomainNotAllowedError,
    PrivateIPError,
    SSRFProtector,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def allowlist():
    """Create a domain allowlist for testing."""
    return DomainAllowlist([
        "api.etherscan.io",
        "*.etherscan.io",
        "api.bscscan.com",
        "*.bscscan.com",
        "example.com",
        "*.example.com",
    ])


@pytest.fixture
def protector(allowlist):
    """Create an SSRF protector for testing."""
    return SSRFProtector(
        allowlist=allowlist,
        require_https=True,
        block_private_ips=True,
    )


@pytest.fixture
def protector_no_private_ip_blocking(allowlist):
    """Create an SSRF protector without private IP blocking."""
    return SSRFProtector(
        allowlist=allowlist,
        require_https=True,
        block_private_ips=False,
    )


# =============================================================================
# DNS Resolution Pinning Tests
# =============================================================================


class TestDNSResolutionPinning:
    """Tests for DNS resolution pinning functionality.
    
    Requirements: 2.8, 2.9
    """

    @pytest.mark.asyncio
    async def test_dns_resolution_cached_within_ttl(self, protector):
        """
        Test that DNS resolution is cached and reused within TTL.
        
        Requirements: 2.8
        """
        hostname = "api.etherscan.io"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            # First resolution
            result1 = await protector._resolve_and_validate(hostname)
            assert result1 == public_ip
            assert mock_resolve.call_count == 1
            
            # Second resolution within TTL should use cache
            result2 = await protector._resolve_and_validate(hostname)
            assert result2 == public_ip
            # Should still be 1 call (cached)
            assert mock_resolve.call_count == 1

    @pytest.mark.asyncio
    async def test_dns_resolution_refreshed_after_ttl(self, protector):
        """
        Test that DNS resolution is refreshed after TTL expires.
        
        Requirements: 2.8
        """
        hostname = "api.etherscan.io"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            # First resolution
            await protector._resolve_and_validate(hostname)
            assert mock_resolve.call_count == 1
            
            # Manually expire the cache
            async with protector._cache_lock:
                if hostname in protector._dns_cache:
                    old_ip, _ = protector._dns_cache[hostname]
                    # Set timestamp to expired (beyond TTL)
                    protector._dns_cache[hostname] = (
                        old_ip,
                        time.time() - protector.DNS_PIN_TTL - 1,
                    )
            
            # Resolution after TTL should make new DNS query
            await protector._resolve_and_validate(hostname)
            assert mock_resolve.call_count == 2

    @pytest.mark.asyncio
    async def test_dns_cache_stores_resolved_ip(self, protector):
        """
        Test that resolved IP is stored in DNS cache.
        
        Requirements: 2.8
        """
        hostname = "api.etherscan.io"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            await protector._resolve_and_validate(hostname)
            
            # Verify cache entry
            async with protector._cache_lock:
                assert hostname in protector._dns_cache
                cached_ip, timestamp = protector._dns_cache[hostname]
                assert cached_ip == public_ip
                assert timestamp <= time.time()

    @pytest.mark.asyncio
    async def test_dns_resolution_retry_on_failure(self, protector):
        """
        Test that DNS resolution retries once on transient failure.
        
        Requirements: 2.8
        """
        import socket
        
        hostname = "api.etherscan.io"
        public_ip = "104.26.10.33"
        
        # Mock socket.gethostbyname to fail first, succeed second
        call_count = 0
        
        def mock_gethostbyname(h):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise socket.gaierror("Temporary failure")
            return public_ip
        
        with patch("socket.gethostbyname", side_effect=mock_gethostbyname):
            # Should succeed after retry
            result = await protector._resolve_and_validate(hostname)
            assert result == public_ip
            assert call_count == 2  # First failed, second succeeded

    @pytest.mark.asyncio
    async def test_dns_resolution_fails_after_retries(self, protector):
        """
        Test that DNS resolution fails after retry attempts exhausted.

        Requirements: 2.8
        """
        hostname = "api.etherscan.io"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.side_effect = DNSResolutionError(
                f"DNS resolution failed for '{hostname}'"
            )
            
            with pytest.raises(DNSResolutionError) as exc_info:
                await protector._resolve_and_validate(hostname)
            
            assert hostname in str(exc_info.value)


# =============================================================================
# DNS Rebinding Detection Tests
# =============================================================================


class TestDNSRebindingDetection:
    """Tests for DNS rebinding attack detection.
    
    Requirements: 2.8, 2.9
    """

    def test_rebinding_detected_public_to_private(self, protector):
        """
        Test that rebinding is detected when IP changes from public to private.
        
        Requirements: 2.9
        """
        domain = "malicious.example.com"
        public_ip = "104.26.10.33"
        private_ip = "192.168.1.1"
        
        result = protector._is_rebinding_attempt(domain, public_ip, private_ip)
        assert result is True

    def test_no_rebinding_when_ip_unchanged(self, protector):
        """
        Test that no rebinding is detected when IP stays the same.
        
        Requirements: 2.9
        """
        domain = "api.etherscan.io"
        public_ip = "104.26.10.33"
        
        result = protector._is_rebinding_attempt(domain, public_ip, public_ip)
        assert result is False

    def test_no_rebinding_public_to_public(self, protector):
        """
        Test that no rebinding is detected when IP changes between public IPs.
        
        Requirements: 2.9
        """
        domain = "api.etherscan.io"
        public_ip1 = "104.26.10.33"
        public_ip2 = "104.26.10.34"
        
        result = protector._is_rebinding_attempt(domain, public_ip1, public_ip2)
        assert result is False

    def test_no_rebinding_private_to_private(self, protector):
        """
        Test that no rebinding is detected when IP changes between private IPs.
        
        Requirements: 2.9
        """
        domain = "internal.example.com"
        private_ip1 = "192.168.1.1"
        private_ip2 = "10.0.0.1"
        
        result = protector._is_rebinding_attempt(domain, private_ip1, private_ip2)
        assert result is False

    def test_no_rebinding_private_to_public(self, protector):
        """
        Test that no rebinding is detected when IP changes from private to public.
        
        This is not considered an attack (private→public is safe).
        
        Requirements: 2.9
        """
        domain = "example.com"
        private_ip = "192.168.1.1"
        public_ip = "104.26.10.33"
        
        result = protector._is_rebinding_attempt(domain, private_ip, public_ip)
        assert result is False

    def test_rebinding_detected_with_loopback(self, protector):
        """
        Test that rebinding is detected when IP changes to loopback.
        
        Requirements: 2.9
        """
        domain = "malicious.example.com"
        public_ip = "104.26.10.33"
        loopback_ip = "127.0.0.1"
        
        result = protector._is_rebinding_attempt(domain, public_ip, loopback_ip)
        assert result is True

    def test_rebinding_detected_with_10_range(self, protector):
        """
        Test that rebinding is detected when IP changes to 10.x.x.x range.
        
        Requirements: 2.9
        """
        domain = "malicious.example.com"
        public_ip = "8.8.8.8"
        private_ip = "10.0.0.1"
        
        result = protector._is_rebinding_attempt(domain, public_ip, private_ip)
        assert result is True

    def test_rebinding_detected_with_172_range(self, protector):
        """
        Test that rebinding is detected when IP changes to 172.16-31.x.x range.
        
        Requirements: 2.9
        """
        domain = "malicious.example.com"
        public_ip = "8.8.8.8"
        private_ip = "172.16.0.1"
        
        result = protector._is_rebinding_attempt(domain, public_ip, private_ip)
        assert result is True


# =============================================================================
# Redirect Validation Tests
# =============================================================================


class TestRedirectValidation:
    """Tests for redirect validation with DNS rebinding protection.
    
    Requirements: 2.8, 2.9
    """

    @pytest.mark.asyncio
    async def test_redirect_to_allowed_domain_passes(self, protector):
        """
        Test that redirect to an allowed domain with public IP passes.
        
        Requirements: 2.8
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://www.etherscan.io/api"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            # Should not raise any exception
            await protector.validate_redirect(original_url, redirect_url)

    @pytest.mark.asyncio
    async def test_redirect_to_disallowed_domain_rejected(self, protector):
        """
        Test that redirect to a disallowed domain is rejected.
        
        Requirements: 2.8
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://malicious.com/api"
        
        with pytest.raises(DomainNotAllowedError):
            await protector.validate_redirect(original_url, redirect_url)

    @pytest.mark.asyncio
    async def test_redirect_to_private_ip_rejected(self, protector):
        """
        Test that redirect resolving to private IP is rejected.
        
        Requirements: 2.9
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://internal.etherscan.io/api"
        private_ip = "192.168.1.1"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = private_ip
            
            with pytest.raises(PrivateIPError):
                await protector.validate_redirect(original_url, redirect_url)

    @pytest.mark.asyncio
    async def test_redirect_dns_rebinding_detected(self, protector):
        """
        Test that DNS rebinding attack during redirect is detected.
        
        Requirements: 2.8, 2.9
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://api.etherscan.io/redirect"
        public_ip = "104.26.10.33"
        private_ip = "192.168.1.1"
        
        # Pre-populate cache with public IP
        async with protector._cache_lock:
            protector._dns_cache["api.etherscan.io"] = (public_ip, time.time())
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            # DNS now resolves to private IP (rebinding attack)
            mock_resolve.return_value = private_ip
            
            with pytest.raises((DNSRebindingError, PrivateIPError)):
                await protector.validate_redirect(original_url, redirect_url)

    @pytest.mark.asyncio
    async def test_redirect_with_original_resolved_ip(self, protector):
        """
        Test redirect validation with original resolved IP provided.
        
        Requirements: 2.8, 2.9
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://api.etherscan.io/redirect"
        original_ip = "104.26.10.33"
        private_ip = "192.168.1.1"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = private_ip
            
            with pytest.raises((DNSRebindingError, PrivateIPError)):
                await protector.validate_redirect(
                    original_url,
                    redirect_url,
                    original_resolved_ip=original_ip,
                )

    @pytest.mark.asyncio
    async def test_redirect_updates_dns_cache(self, protector):
        """
        Test that successful redirect validation updates DNS cache.
        
        Requirements: 2.8
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://www.etherscan.io/api"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            await protector.validate_redirect(original_url, redirect_url)
            
            # Verify cache was updated
            async with protector._cache_lock:
                assert "www.etherscan.io" in protector._dns_cache
                cached_ip, _ = protector._dns_cache["www.etherscan.io"]
                assert cached_ip == public_ip

    @pytest.mark.asyncio
    async def test_redirect_without_private_ip_blocking(
        self, protector_no_private_ip_blocking
    ):
        """
        Test that redirect validation skips IP checks when disabled.
        
        Requirements: 2.8
        """
        protector = protector_no_private_ip_blocking
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://www.etherscan.io/api"
        
        # Should not call _resolve_hostname when private IP blocking is disabled
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            await protector.validate_redirect(original_url, redirect_url)
            mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_redirect_without_hostname_rejected(self, protector):
        """
        Test that redirect URL without hostname is rejected.
        
        Requirements: 2.8
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https:///path"
        
        with pytest.raises(DomainNotAllowedError):
            await protector.validate_redirect(original_url, redirect_url)

    @pytest.mark.asyncio
    async def test_redirect_to_http_rejected_when_https_required(self, protector):
        """
        Test that redirect to HTTP is rejected when HTTPS is required.
        
        Requirements: 2.8
        """
        from core.security.ssrf_protector import ProtocolNotAllowedError
        
        original_url = "https://api.etherscan.io/api"
        redirect_url = "http://www.etherscan.io/api"
        
        with pytest.raises(ProtocolNotAllowedError):
            await protector.validate_redirect(original_url, redirect_url)

    @pytest.mark.asyncio
    async def test_redirect_cross_domain_allowed(self, protector):
        """
        Test that redirect to different allowed domain passes.
        
        Requirements: 2.8
        """
        original_url = "https://api.etherscan.io/api"
        redirect_url = "https://api.bscscan.com/api"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            # Should not raise any exception
            await protector.validate_redirect(original_url, redirect_url)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDNSRebindingIntegration:
    """Integration tests for DNS rebinding protection.
    
    Requirements: 2.8, 2.9
    """

    @pytest.mark.asyncio
    async def test_full_request_validation_with_dns_pinning(self, protector):
        """
        Test full request validation flow with DNS pinning.
        
        Requirements: 2.8, 2.9
        """
        url = "https://api.etherscan.io/api"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            # First request
            await protector.validate_request(url)
            assert mock_resolve.call_count == 1
            
            # Second request should use cached DNS
            await protector.validate_request(url)
            assert mock_resolve.call_count == 1  # Still 1, used cache

    @pytest.mark.asyncio
    async def test_concurrent_requests_thread_safe(self, protector):
        """
        Test that concurrent requests are thread-safe with DNS cache.
        
        Requirements: 2.8
        """
        url = "https://api.etherscan.io/api"
        public_ip = "104.26.10.33"
        
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            
            # Run multiple concurrent validations
            tasks = [
                protector.validate_request(url)
                for _ in range(10)
            ]
            await asyncio.gather(*tasks)
            
            # Should have resolved at least once
            assert mock_resolve.call_count >= 1
            # But not 10 times due to caching
            assert mock_resolve.call_count < 10

    @pytest.mark.asyncio
    async def test_rebinding_attack_scenario(self, protector):
        """
        Test a realistic DNS rebinding attack scenario.
        
        Scenario:
        1. Initial request resolves to public IP
        2. Attacker changes DNS to point to private IP
        3. Redirect should be blocked
        
        Requirements: 2.8, 2.9
        """
        hostname = "attacker.example.com"
        original_url = f"https://{hostname}/api"
        redirect_url = f"https://{hostname}/internal"
        public_ip = "104.26.10.33"
        private_ip = "192.168.1.1"
        
        # Add attacker domain to allowlist for this test
        protector._allowlist._exact_domains.add(hostname)
        
        # First resolution returns public IP
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = public_ip
            await protector.validate_request(original_url)
        
        # Now attacker changes DNS to private IP
        with patch.object(
            protector, "_resolve_hostname", new_callable=AsyncMock
        ) as mock_resolve:
            mock_resolve.return_value = private_ip
            
            # Redirect should be blocked due to rebinding detection
            with pytest.raises((DNSRebindingError, PrivateIPError)):
                await protector.validate_redirect(
                    original_url,
                    redirect_url,
                    original_resolved_ip=public_ip,
                )
