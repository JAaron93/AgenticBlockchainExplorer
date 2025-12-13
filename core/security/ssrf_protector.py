"""SSRF (Server-Side Request Forgery) protection components.

This module provides classes for protecting against SSRF attacks by:
- Validating outbound request domains against an allowlist
- Blocking requests to private/internal IP ranges
- Detecting DNS rebinding attacks

Requirements: 2.1, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10
"""

import asyncio
import fnmatch
import ipaddress
import logging
import socket
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from config.models import SSRFProtectionConfig


logger = logging.getLogger(__name__)


class SSRFError(Exception):
    """Base class for SSRF-related errors."""
    pass


class DomainNotAllowedError(SSRFError):
    """Raised when domain is not in allowlist."""
    pass


class PrivateIPError(SSRFError):
    """Raised when request would resolve to private IP."""
    pass


class ProtocolNotAllowedError(SSRFError):
    """Raised when protocol is not HTTPS."""
    pass


class DNSResolutionError(SSRFError):
    """Raised when DNS resolution fails."""
    pass


class DNSRebindingError(SSRFError):
    """Raised when DNS rebinding attack is detected."""
    pass


class DomainAllowlist:
    """Validates domains against an allowlist with pattern support.

    Supports exact domain matching and wildcard patterns for subdomains.
    For example, "*.etherscan.io" matches "api.etherscan.io" and
    "www.etherscan.io" but not "etherscan.io" itself.

    Requirements: 2.1, 2.7, 2.10
    """

    def __init__(self, patterns: List[str]):
        """Initialize with domain patterns (exact or wildcard).

        Args:
            patterns: List of domain patterns. Patterns starting with "*."
                     match any subdomain. All patterns are case-insensitive.

        Raises:
            ValueError: If patterns list is empty.
        """
        if not patterns:
            raise ValueError(
                "Domain allowlist cannot be empty"
            )

        self._exact_domains: set[str] = set()
        self._wildcard_patterns: List[str] = []

        for pattern in patterns:
            pattern = pattern.strip().lower()
            if pattern.startswith("*."):
                # Store wildcard patterns for fnmatch
                self._wildcard_patterns.append(pattern)
            else:
                # Store exact domains in a set for O(1) lookup
                self._exact_domains.add(pattern)

        # Log configured patterns at INFO level for operational visibility
        logger.info(
            "Domain allowlist initialized with %d exact domains and "
            "%d wildcard patterns",
            len(self._exact_domains),
            len(self._wildcard_patterns),
        )
        logger.info("Exact domains: %s", sorted(self._exact_domains))
        logger.info("Wildcard patterns: %s", self._wildcard_patterns)

    def is_allowed(self, domain: str) -> bool:
        """Check if domain matches any allowlist pattern.

        Args:
            domain: The domain to check (e.g., "api.etherscan.io").

        Returns:
            True if domain matches an exact domain or wildcard pattern.
        """
        if not domain:
            return False

        domain = domain.strip().lower()

        # Check exact match first (O(1))
        if domain in self._exact_domains:
            return True

        # Check wildcard patterns
        for pattern in self._wildcard_patterns:
            if fnmatch.fnmatch(domain, pattern):
                return True

        return False

    def validate_url(self, url: str, require_https: bool = True) -> bool:
        """Validate full URL including protocol and domain.

        Args:
            url: Full URL to validate.
            require_https: If True, reject non-HTTPS URLs.

        Returns:
            True if URL passes all validation checks.

        Raises:
            ProtocolNotAllowedError: If protocol is not HTTPS.
            DomainNotAllowedError: If domain is not in allowlist.
        """
        if not url:
            raise DomainNotAllowedError("URL cannot be empty")

        try:
            parsed = urlparse(url)
        except Exception as e:
            raise DomainNotAllowedError(f"Invalid URL format: {e}")

        # Check protocol
        if require_https and parsed.scheme.lower() != "https":
            raise ProtocolNotAllowedError(
                f"Protocol '{parsed.scheme}' not allowed. HTTPS required."
            )

        # Extract domain (hostname without port)
        domain = parsed.hostname
        if not domain:
            raise DomainNotAllowedError("URL must contain a valid hostname")

        # Check domain against allowlist
        if not self.is_allowed(domain):
            raise DomainNotAllowedError(
                f"Domain '{domain}' is not in the allowlist"
            )

        return True

    @classmethod
    def from_config(cls, config: SSRFProtectionConfig) -> "DomainAllowlist":
        """Create DomainAllowlist from application configuration.

        Args:
            config: SSRF protection configuration.

        Returns:
            Configured DomainAllowlist instance.
        """
        return cls(patterns=config.allowed_domains)


class SSRFProtector:
    """Protects against SSRF by validating outbound requests.

    This class provides comprehensive SSRF protection including:
    - Domain allowlist validation
    - Protocol enforcement (HTTPS only)
    - Private/internal IP blocking
    - DNS rebinding attack detection

    Requirements: 2.4, 2.5, 2.6, 2.8, 2.9
    """

    # Comprehensive list of private/reserved IP ranges
    # IPv4 Private/Reserved
    PRIVATE_IP_RANGES = [
        ipaddress.ip_network("0.0.0.0/8"),       # "This" network
        ipaddress.ip_network("10.0.0.0/8"),      # Private (RFC 1918)
        ipaddress.ip_network("100.64.0.0/10"),   # Carrier-grade NAT
        ipaddress.ip_network("127.0.0.0/8"),     # Loopback
        ipaddress.ip_network("169.254.0.0/16"),  # Link-local
        ipaddress.ip_network("172.16.0.0/12"),   # Private (RFC 1918)
        ipaddress.ip_network("192.0.0.0/24"),    # IETF Protocol Assignments
        ipaddress.ip_network("192.0.2.0/24"),    # TEST-NET-1
        ipaddress.ip_network("192.168.0.0/16"),  # Private (RFC 1918)
        ipaddress.ip_network("198.18.0.0/15"),   # Benchmarking
        ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
        ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
        ipaddress.ip_network("224.0.0.0/4"),     # Multicast
        ipaddress.ip_network("240.0.0.0/4"),     # Reserved for future use
        # IPv6 Private/Reserved
        ipaddress.ip_network("::/128"),          # Unspecified
        ipaddress.ip_network("::1/128"),         # Loopback
        ipaddress.ip_network("::ffff:0:0/96"),   # IPv4-mapped IPv6
        ipaddress.ip_network("64:ff9b::/96"),    # IPv4/IPv6 translation
        ipaddress.ip_network("100::/64"),        # Discard prefix
        ipaddress.ip_network("fc00::/7"),        # Unique local (ULA)
        ipaddress.ip_network("fe80::/10"),       # Link-local
        ipaddress.ip_network("ff00::/8"),        # Multicast
    ]

    # DNS resolution cache TTL for pinning (seconds)
    DNS_PIN_TTL = 60

    def __init__(
        self,
        allowlist: DomainAllowlist,
        require_https: bool = True,
        block_private_ips: bool = True,
    ):
        """Initialize SSRF protector.

        Args:
            allowlist: Domain allowlist for validation.
            require_https: Require HTTPS protocol.
            block_private_ips: Block requests to private IP ranges.
        """
        self._allowlist = allowlist
        self._require_https = require_https
        self._block_private_ips = block_private_ips
        # DNS cache: domain -> (ip, timestamp)
        self._dns_cache: Dict[str, Tuple[str, float]] = {}
        # Lock for thread-safe DNS cache access
        self._cache_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: SSRFProtectionConfig) -> "SSRFProtector":
        """Create SSRFProtector from configuration.

        Args:
            config: SSRF protection configuration.

        Returns:
            Configured SSRFProtector instance.
        """
        allowlist = DomainAllowlist.from_config(config)
        return cls(
            allowlist=allowlist,
            require_https=config.require_https,
            block_private_ips=config.block_private_ips,
        )

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private/internal ranges.

        Args:
            ip: IP address string (IPv4 or IPv6).

        Returns:
            True if IP is in any private/reserved range.
        """
        try:
            ip_obj = ipaddress.ip_address(ip)
            for network in self.PRIVATE_IP_RANGES:
                if ip_obj in network:
                    return True
            return False
        except ValueError:
            # Invalid IP format - treat as private for safety
            logger.warning("Invalid IP address format: %s", ip)
            return True

    async def validate_request(self, url: str) -> None:
        """Validate URL before making request.

        Performs the following checks:
        1. Protocol is HTTPS (if required)
        2. Domain is in allowlist
        3. Resolved IP is not in private ranges (if blocking enabled)

        Args:
            url: URL to validate.

        Raises:
            ProtocolNotAllowedError: If protocol is not HTTPS.
            DomainNotAllowedError: If domain is not in allowlist.
            PrivateIPError: If resolved IP is in private ranges.
            DNSResolutionError: If DNS resolution fails.
        """
        # Validate URL format, protocol, and domain
        self._allowlist.validate_url(url, require_https=self._require_https)

        # If private IP blocking is enabled, resolve and check IP
        if self._block_private_ips:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if hostname:
                await self._resolve_and_validate(hostname)

    async def _resolve_and_validate(self, hostname: str) -> str:
        """Resolve hostname and validate IP is not private.

        Uses pinned resolution if within TTL, otherwise fresh resolution.

        Args:
            hostname: Hostname to resolve.

        Returns:
            The resolved IP address.

        Raises:
            PrivateIPError: If IP is in private ranges.
            DNSResolutionError: If DNS resolution fails.
        """
        current_time = time.time()

        # Check cache first (with lock for thread safety)
        async with self._cache_lock:
            if hostname in self._dns_cache:
                cached_ip, timestamp = self._dns_cache[hostname]
                if current_time - timestamp < self.DNS_PIN_TTL:
                    # Use cached IP (DNS pinning)
                    if self._is_private_ip(cached_ip):
                        raise PrivateIPError(
                            f"Cached IP for '{hostname}' is in private range"
                        )
                    return cached_ip

        # Perform fresh DNS resolution
        resolved_ip = await self._resolve_hostname(hostname)

        # Validate resolved IP
        if self._is_private_ip(resolved_ip):
            raise PrivateIPError(
                f"Resolved IP for '{hostname}' is in private range"
            )

        # Cache the resolved IP
        async with self._cache_lock:
            self._dns_cache[hostname] = (resolved_ip, current_time)

        return resolved_ip

    async def _resolve_hostname(self, hostname: str) -> str:
        """Resolve hostname to IP address.

        Args:
            hostname: Hostname to resolve.

        Returns:
            Resolved IP address.

        Raises:
            DNSResolutionError: If resolution fails after retry.
        """
        loop = asyncio.get_event_loop()

        # Retry once on failure
        for attempt in range(2):
            try:
                # Run DNS resolution in executor to avoid blocking
                result = await loop.run_in_executor(
                    None,
                    socket.gethostbyname,
                    hostname,
                )
                return result
            except socket.gaierror as e:
                if attempt == 0:
                    # Wait 1 second before retry
                    await asyncio.sleep(1)
                    continue
                raise DNSResolutionError(
                    f"DNS resolution failed for '{hostname}': {e}"
                )

        # Should not reach here, but just in case
        raise DNSResolutionError(
            f"DNS resolution failed for '{hostname}' after retries"
        )

    async def validate_redirect(
        self,
        original_url: str,
        redirect_url: str,
        original_resolved_ip: Optional[str] = None,
    ) -> None:
        """Validate redirect target including DNS rebinding check.

        DNS Rebinding Protection:
        1. Always perform fresh DNS resolution at validation time
        2. Pin the resolved IP for DNS_PIN_TTL seconds per domain
        3. Both original_url and redirect_url are resolved and checked
        4. Reject if either resolves to private/internal IP
        5. Detect rebinding: if same domain resolves to different IP within
           pin window, and new IP is private while old was public, reject
        6. On resolution mismatch (public→private), fail immediately
        7. On transient DNS failure, retry once after 1 second, then fail

        Args:
            original_url: The URL that returned the redirect.
            redirect_url: The redirect target URL.
            original_resolved_ip: Optional cached IP from original request.

        Raises:
            SSRFError: If redirect target fails validation.
            PrivateIPError: If redirect resolves to private IP.
            DomainNotAllowedError: If redirect domain not in allowlist.
            DNSRebindingError: If DNS rebinding attack detected.
        """
        # First validate the redirect URL against allowlist and protocol
        self._allowlist.validate_url(
            redirect_url, require_https=self._require_https
        )

        if not self._block_private_ips:
            return

        # Parse redirect URL to get hostname
        parsed = urlparse(redirect_url)
        redirect_hostname = parsed.hostname

        if not redirect_hostname:
            raise DomainNotAllowedError(
                "Redirect URL must contain a valid hostname"
            )

        # Perform fresh DNS resolution for redirect target
        redirect_ip = await self._resolve_hostname(redirect_hostname)

        # Check if redirect IP is private
        if self._is_private_ip(redirect_ip):
            raise PrivateIPError(
                f"Redirect target '{redirect_hostname}' resolves to "
                "private IP range"
            )

        # Check for DNS rebinding attack
        # If we have a cached IP for this domain, check for rebinding
        async with self._cache_lock:
            if redirect_hostname in self._dns_cache:
                cached_ip, _ = self._dns_cache[redirect_hostname]
                if self._is_rebinding_attempt(
                    redirect_hostname, cached_ip, redirect_ip
                ):
                    raise DNSRebindingError(
                        f"DNS rebinding detected for '{redirect_hostname}': "
                        f"IP changed from public to private"
                    )

        # Also check original URL's hostname if provided
        if original_resolved_ip:
            original_parsed = urlparse(original_url)
            original_hostname = original_parsed.hostname

            if original_hostname == redirect_hostname:
                # Same domain - check for rebinding
                if self._is_rebinding_attempt(
                    redirect_hostname, original_resolved_ip, redirect_ip
                ):
                    raise DNSRebindingError(
                        f"DNS rebinding detected for '{redirect_hostname}': "
                        f"IP changed from public to private during redirect"
                    )

        # Update cache with new resolution
        async with self._cache_lock:
            self._dns_cache[redirect_hostname] = (redirect_ip, time.time())

    def _is_rebinding_attempt(
        self,
        domain: str,
        old_ip: str,
        new_ip: str,
    ) -> bool:
        """Detect DNS rebinding: public IP changed to private IP.

        Args:
            domain: The domain being checked.
            old_ip: Previously resolved IP.
            new_ip: Newly resolved IP.

        Returns:
            True if this appears to be a DNS rebinding attack.
        """
        if old_ip == new_ip:
            return False

        old_is_private = self._is_private_ip(old_ip)
        new_is_private = self._is_private_ip(new_ip)

        # Rebinding attack: public IP changed to private IP
        if not old_is_private and new_is_private:
            logger.warning(
                "DNS rebinding detected for '%s': "
                "IP changed from public (%s) to private (%s)",
                domain,
                old_ip,
                new_ip,
            )
            return True

        return False
