"""
Property-based tests for SSRF protection domain allowlist enforcement.

These tests use Hypothesis to verify correctness properties defined in the design document.

**Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
**Validates: Requirements 2.4, 2.5, 2.6**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume

from core.security.ssrf_protector import (
    DomainAllowlist,
    DomainNotAllowedError,
    ProtocolNotAllowedError,
)


# =============================================================================
# Test Data Strategies
# =============================================================================


def valid_domain_label():
    """Generate a valid DNS label (part of a domain name).
    
    DNS labels must:
    - Be 1-63 characters
    - Start with a letter
    - Contain only letters, digits, and hyphens
    - Not end with a hyphen
    """
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
        min_size=1,
        max_size=20,
    ).filter(lambda x: x[0].isalpha() if x else False)


def valid_tld():
    """Generate a valid top-level domain."""
    return st.sampled_from([
        "com", "io", "org", "net", "co", "dev", "app", "xyz"
    ])


@st.composite
def valid_domain(draw):
    """Generate a valid domain name."""
    # Generate 1-3 subdomains
    num_labels = draw(st.integers(min_value=1, max_value=3))
    labels = [draw(valid_domain_label()) for _ in range(num_labels)]
    tld = draw(valid_tld())
    return ".".join(labels + [tld])


@st.composite
def domain_with_subdomain(draw, base_domain: str):
    """Generate a subdomain of a given base domain."""
    subdomain = draw(valid_domain_label())
    return f"{subdomain}.{base_domain}"


def https_url_for_domain(domain: str) -> str:
    """Create an HTTPS URL for a domain."""
    return f"https://{domain}/api"


def http_url_for_domain(domain: str) -> str:
    """Create an HTTP URL for a domain."""
    return f"http://{domain}/api"


@st.composite
def allowlist_and_matching_domain(draw):
    """Generate an allowlist and a domain that should match it.
    
    Returns: (allowlist_patterns, domain, match_type)
    where match_type is 'exact' or 'wildcard'
    """
    # Decide if we're testing exact match or wildcard match
    match_type = draw(st.sampled_from(["exact", "wildcard"]))
    
    base_domain = draw(valid_domain())
    
    if match_type == "exact":
        # Exact match: domain is in allowlist directly
        patterns = [base_domain]
        domain = base_domain
    else:
        # Wildcard match: allowlist has *.base_domain, domain is subdomain.base_domain
        patterns = [f"*.{base_domain}"]
        subdomain = draw(valid_domain_label())
        domain = f"{subdomain}.{base_domain}"
    
    # Add some other random patterns to make it more realistic
    extra_patterns = draw(st.lists(valid_domain(), min_size=0, max_size=3))
    all_patterns = patterns + extra_patterns
    
    return all_patterns, domain, match_type


@st.composite
def allowlist_and_non_matching_domain(draw):
    """Generate an allowlist and a domain that should NOT match it.
    
    Returns: (allowlist_patterns, non_matching_domain)
    """
    # Generate allowlist patterns
    patterns = draw(st.lists(valid_domain(), min_size=1, max_size=5))
    
    # Generate a domain that is NOT in the allowlist
    non_matching = draw(valid_domain())
    
    # Ensure it doesn't match any pattern
    assume(non_matching not in patterns)
    assume(not any(
        non_matching.endswith(p.lstrip("*.")) 
        for p in patterns 
        if p.startswith("*.")
    ))
    
    return patterns, non_matching


# =============================================================================
# Property Tests
# =============================================================================


class TestDomainAllowlistEnforcement:
    """
    Property tests for domain allowlist enforcement.
    
    **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
    
    For any URL, the SSRF protector SHALL allow the request if and only if
    the domain matches an allowlist pattern AND the protocol is HTTPS.
    
    **Validates: Requirements 2.4, 2.5, 2.6**
    """

    @settings(max_examples=100)
    @given(data=allowlist_and_matching_domain())
    def test_property_2_allowed_domain_with_https_passes(self, data):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any domain that matches an allowlist pattern, validate_url()
        SHALL return True when the protocol is HTTPS.
        
        **Validates: Requirements 2.4, 2.5**
        """
        patterns, domain, match_type = data
        allowlist = DomainAllowlist(patterns)
        
        # Create HTTPS URL
        url = https_url_for_domain(domain)
        
        # Should pass validation (returns True, no exception)
        result = allowlist.validate_url(url, require_https=True)
        assert result is True, (
            f"validate_url() should return True for allowed domain '{domain}' "
            f"with HTTPS. Patterns: {patterns}, Match type: {match_type}"
        )

    @settings(max_examples=100)
    @given(data=allowlist_and_matching_domain())
    def test_property_2_allowed_domain_with_http_rejected_when_https_required(self, data):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any domain that matches an allowlist pattern, validate_url()
        SHALL reject the request when protocol is HTTP and HTTPS is required.
        
        **Validates: Requirements 2.5**
        """
        patterns, domain, match_type = data
        allowlist = DomainAllowlist(patterns)
        
        # Create HTTP URL (not HTTPS)
        url = http_url_for_domain(domain)
        
        # Should raise ProtocolNotAllowedError
        with pytest.raises(ProtocolNotAllowedError) as exc_info:
            allowlist.validate_url(url, require_https=True)
        
        assert "HTTPS required" in str(exc_info.value), (
            f"Error message should mention HTTPS requirement. Got: {exc_info.value}"
        )

    @settings(max_examples=100)
    @given(data=allowlist_and_non_matching_domain())
    def test_property_2_non_allowed_domain_rejected(self, data):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any domain that does NOT match an allowlist pattern, validate_url()
        SHALL reject the request regardless of protocol.
        
        **Validates: Requirements 2.4, 2.6**
        """
        patterns, non_matching_domain = data
        allowlist = DomainAllowlist(patterns)
        
        # Create HTTPS URL with non-matching domain
        url = https_url_for_domain(non_matching_domain)
        
        # Should raise DomainNotAllowedError
        with pytest.raises(DomainNotAllowedError) as exc_info:
            allowlist.validate_url(url, require_https=True)
        
        assert "not in the allowlist" in str(exc_info.value), (
            f"Error message should mention domain not in allowlist. Got: {exc_info.value}"
        )

    @settings(max_examples=100)
    @given(domain=valid_domain())
    def test_property_2_exact_match_allows_domain(self, domain):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any domain in the allowlist as an exact match, is_allowed()
        SHALL return True for that exact domain.
        
        **Validates: Requirements 2.4**
        """
        allowlist = DomainAllowlist([domain])
        
        assert allowlist.is_allowed(domain) is True, (
            f"is_allowed() should return True for exact match domain '{domain}'"
        )

    @settings(max_examples=100)
    @given(
        base_domain=valid_domain(),
        subdomain_label=valid_domain_label(),
    )
    def test_property_2_wildcard_match_allows_subdomains(self, base_domain, subdomain_label):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any wildcard pattern *.base_domain in the allowlist, is_allowed()
        SHALL return True for any subdomain.base_domain.
        
        **Validates: Requirements 2.4**
        """
        wildcard_pattern = f"*.{base_domain}"
        allowlist = DomainAllowlist([wildcard_pattern])
        
        subdomain = f"{subdomain_label}.{base_domain}"
        
        assert allowlist.is_allowed(subdomain) is True, (
            f"is_allowed() should return True for subdomain '{subdomain}' "
            f"matching wildcard pattern '{wildcard_pattern}'"
        )

    @settings(max_examples=100)
    @given(base_domain=valid_domain())
    def test_property_2_wildcard_does_not_match_base_domain(self, base_domain):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any wildcard pattern *.base_domain in the allowlist, is_allowed()
        SHALL return False for the base domain itself (without subdomain).
        
        **Validates: Requirements 2.4**
        """
        wildcard_pattern = f"*.{base_domain}"
        allowlist = DomainAllowlist([wildcard_pattern])
        
        # The base domain itself should NOT match the wildcard
        assert allowlist.is_allowed(base_domain) is False, (
            f"is_allowed() should return False for base domain '{base_domain}' "
            f"when only wildcard pattern '{wildcard_pattern}' is in allowlist"
        )

    @settings(max_examples=100)
    @given(
        domain1=valid_domain(),
        domain2=valid_domain(),
    )
    def test_property_2_domain_case_insensitivity(self, domain1, domain2):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any domain, the allowlist check SHALL be case-insensitive.
        
        **Validates: Requirements 2.4**
        """
        assume(domain1 != domain2)  # Ensure different domains
        
        # Add domain1 in lowercase
        allowlist = DomainAllowlist([domain1.lower()])
        
        # Should match regardless of case
        assert allowlist.is_allowed(domain1.upper()) is True, (
            f"is_allowed() should be case-insensitive. "
            f"'{domain1.upper()}' should match '{domain1.lower()}'"
        )
        assert allowlist.is_allowed(domain1.lower()) is True, (
            f"is_allowed() should match lowercase domain"
        )

    @settings(max_examples=100)
    @given(data=allowlist_and_matching_domain())
    def test_property_2_http_allowed_when_https_not_required(self, data):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any allowed domain, validate_url() SHALL accept HTTP when
        require_https is False.
        
        **Validates: Requirements 2.5**
        """
        patterns, domain, match_type = data
        allowlist = DomainAllowlist(patterns)
        
        # Create HTTP URL
        url = http_url_for_domain(domain)
        
        # Should pass when HTTPS is not required
        result = allowlist.validate_url(url, require_https=False)
        assert result is True, (
            f"validate_url() should return True for HTTP URL when "
            f"require_https=False. Domain: '{domain}'"
        )

    @settings(max_examples=100)
    @given(
        patterns=st.lists(valid_domain(), min_size=1, max_size=5),
    )
    def test_property_2_empty_url_rejected(self, patterns):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any allowlist, validate_url() SHALL reject empty URLs.
        
        **Validates: Requirements 2.4**
        """
        allowlist = DomainAllowlist(patterns)
        
        with pytest.raises(DomainNotAllowedError):
            allowlist.validate_url("", require_https=True)

    @settings(max_examples=100)
    @given(
        patterns=st.lists(valid_domain(), min_size=1, max_size=5),
    )
    def test_property_2_url_without_hostname_rejected(self, patterns):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any allowlist, validate_url() SHALL reject URLs without a valid hostname.
        
        **Validates: Requirements 2.4**
        """
        allowlist = DomainAllowlist(patterns)
        
        # URL without hostname
        with pytest.raises(DomainNotAllowedError):
            allowlist.validate_url("https:///path", require_https=True)

    def test_property_2_empty_allowlist_rejected(self):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        Creating a DomainAllowlist with an empty pattern list SHALL raise ValueError.
        
        **Validates: Requirements 2.6**
        """
        with pytest.raises(ValueError) as exc_info:
            DomainAllowlist([])
        
        assert "cannot be empty" in str(exc_info.value), (
            f"Error message should mention empty allowlist. Got: {exc_info.value}"
        )

    @settings(max_examples=100)
    @given(
        domain=valid_domain(),
        path=st.text(alphabet="abcdefghijklmnopqrstuvwxyz/", min_size=0, max_size=20),
        query=st.text(alphabet="abcdefghijklmnopqrstuvwxyz=&", min_size=0, max_size=20),
    )
    def test_property_2_url_path_and_query_do_not_affect_domain_check(
        self, domain, path, query
    ):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any URL, the domain check SHALL only consider the hostname,
        not the path or query parameters.
        
        **Validates: Requirements 2.4**
        """
        allowlist = DomainAllowlist([domain])
        
        # Build URL with path and query
        url = f"https://{domain}"
        if path:
            url += f"/{path.lstrip('/')}"
        if query:
            url += f"?{query}"
        
        # Should still pass validation
        result = allowlist.validate_url(url, require_https=True)
        assert result is True, (
            f"validate_url() should pass for allowed domain regardless of "
            f"path/query. URL: {url}"
        )

    @settings(max_examples=100)
    @given(
        domain=valid_domain(),
        port=st.integers(min_value=1, max_value=65535),
    )
    def test_property_2_port_does_not_affect_domain_check(self, domain, port):
        """
        **Feature: agent-security-hardening, Property 2: Domain Allowlist Enforcement**
        
        For any URL with a port number, the domain check SHALL only consider
        the hostname, not the port.
        
        **Validates: Requirements 2.4**
        """
        allowlist = DomainAllowlist([domain])
        
        # URL with port
        url = f"https://{domain}:{port}/api"
        
        # Should still pass validation
        result = allowlist.validate_url(url, require_https=True)
        assert result is True, (
            f"validate_url() should pass for allowed domain regardless of port. "
            f"URL: {url}"
        )


# =============================================================================
# Property 3: Private IP Blocking - Strategies
# =============================================================================


def ipv4_10_range():
    """Generate IPs in 10.0.0.0/8 (Private RFC 1918)."""
    return st.tuples(
        st.just(10),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_172_16_range():
    """Generate IPs in 172.16.0.0/12 (Private RFC 1918)."""
    return st.tuples(
        st.just(172),
        st.integers(min_value=16, max_value=31),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_192_168_range():
    """Generate IPs in 192.168.0.0/16 (Private RFC 1918)."""
    return st.tuples(
        st.just(192),
        st.just(168),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_127_range():
    """Generate IPs in 127.0.0.0/8 (Loopback)."""
    return st.tuples(
        st.just(127),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_169_254_range():
    """Generate IPs in 169.254.0.0/16 (Link-local)."""
    return st.tuples(
        st.just(169),
        st.just(254),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_100_64_range():
    """Generate IPs in 100.64.0.0/10 (Carrier-grade NAT)."""
    return st.tuples(
        st.just(100),
        st.integers(min_value=64, max_value=127),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_0_range():
    """Generate IPs in 0.0.0.0/8 ('This' network)."""
    return st.tuples(
        st.just(0),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_multicast_range():
    """Generate IPs in 224.0.0.0/4 (Multicast)."""
    return st.tuples(
        st.integers(min_value=224, max_value=239),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")

def ipv4_reserved_range():
    """Generate IPs in 240.0.0.0/4 (Reserved for future use)."""
    return st.tuples(
        st.integers(min_value=240, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
        st.integers(min_value=0, max_value=255),
    ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}")


def ipv6_loopback():
    """Generate IPv6 loopback address (::1)."""
    return st.just("::1")


def ipv6_unspecified():
    """Generate IPv6 unspecified address (::)."""
    return st.just("::")


def ipv6_link_local():
    """Generate IPs in fe80::/10 (Link-local)."""
    return st.tuples(
        st.integers(min_value=0xfe80, max_value=0xfebf),
        st.integers(min_value=0, max_value=0xffff),
        st.integers(min_value=0, max_value=0xffff),
        st.integers(min_value=0, max_value=0xffff),
    ).map(lambda t: f"{t[0]:x}:{t[1]:x}:{t[2]:x}:{t[3]:x}::1")


def ipv6_unique_local():
    """Generate IPs in fc00::/7 (Unique local address)."""
    return st.tuples(
        st.integers(min_value=0xfc00, max_value=0xfdff),
        st.integers(min_value=0, max_value=0xffff),
        st.integers(min_value=0, max_value=0xffff),
        st.integers(min_value=0, max_value=0xffff),
    ).map(lambda t: f"{t[0]:x}:{t[1]:x}:{t[2]:x}:{t[3]:x}::1")


def ipv6_multicast():
    """Generate IPs in ff00::/8 (Multicast)."""
    return st.tuples(
        st.integers(min_value=0xff00, max_value=0xffff),
        st.integers(min_value=0, max_value=0xffff),
        st.integers(min_value=0, max_value=0xffff),
        st.integers(min_value=0, max_value=0xffff),
    ).map(lambda t: f"{t[0]:x}:{t[1]:x}:{t[2]:x}:{t[3]:x}::1")


def any_private_ipv4():
    """Generate any private/reserved IPv4 address."""
    return st.one_of(
        ipv4_10_range(),
        ipv4_172_16_range(),
        ipv4_192_168_range(),
        ipv4_127_range(),
        ipv4_169_254_range(),
        ipv4_100_64_range(),
        ipv4_0_range(),
        ipv4_multicast_range(),
        ipv4_reserved_range(),
    )


def any_private_ipv6():
    """Generate any private/reserved IPv6 address."""
    return st.one_of(
        ipv6_loopback(),
        ipv6_unspecified(),
        ipv6_link_local(),
        ipv6_unique_local(),
        ipv6_multicast(),
    )


def any_private_ip():
    """Generate any private/reserved IP address (IPv4 or IPv6)."""
    return st.one_of(
        any_private_ipv4(),
        any_private_ipv6(),
    )


def public_ipv4():
    """Generate a public IPv4 address.

    Generates IPs in ranges that are NOT private/reserved.
    Uses common public IP ranges like 8.x.x.x, 1.x.x.x, etc.
    """
    return st.one_of(
        # 1.0.0.0 - 9.255.255.255 (excluding 0.x.x.x)
        st.tuples(
            st.integers(min_value=1, max_value=9),
            st.integers(min_value=0, max_value=255),
            st.integers(min_value=0, max_value=255),
            st.integers(min_value=0, max_value=255),
        ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}"),
        # 11.0.0.0 - 99.255.255.255 (excluding 10.x.x.x)
        st.tuples(
            st.integers(min_value=11, max_value=99),
            st.integers(min_value=0, max_value=255),
            st.integers(min_value=0, max_value=255),
            st.integers(min_value=0, max_value=255),
        ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}"),
        # 128.0.0.0 - 168.255.255.255 (safe public range)
        st.tuples(
            st.integers(min_value=128, max_value=168),
            st.integers(min_value=0, max_value=255),
            st.integers(min_value=0, max_value=255),
            st.integers(min_value=0, max_value=255),
        ).map(lambda t: f"{t[0]}.{t[1]}.{t[2]}.{t[3]}"),
    )


# =============================================================================
# Property 3: Private IP Blocking - Test Class
# =============================================================================


class TestPrivateIPBlocking:
    """
    Property tests for private IP blocking.

    **Feature: agent-security-hardening, Property 3: Private IP Blocking**

    For any IP address in private ranges (10.x, 172.16-31.x, 192.168.x,
    127.x, ::1), the SSRF protector SHALL block requests resolving to that IP.

    **Validates: Requirements 2.9**
    """

    @settings(max_examples=100)
    @given(private_ip=any_private_ipv4())
    def test_property_3_private_ipv4_detected(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IPv4 address in private/reserved ranges, _is_private_ip()
        SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        result = protector._is_private_ip(private_ip)
        assert result is True, (
            f"_is_private_ip() should return True for private IPv4 "
            f"'{private_ip}'"
        )

    @settings(max_examples=100)
    @given(private_ip=any_private_ipv6())
    def test_property_3_private_ipv6_detected(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IPv6 address in private/reserved ranges, _is_private_ip()
        SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        result = protector._is_private_ip(private_ip)
        assert result is True, (
            f"_is_private_ip() should return True for private IPv6 "
            f"'{private_ip}'"
        )

    @settings(max_examples=100)
    @given(public_ip=public_ipv4())
    def test_property_3_public_ipv4_not_blocked(self, public_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any public IPv4 address, _is_private_ip() SHALL return False.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        result = protector._is_private_ip(public_ip)
        assert result is False, (
            f"_is_private_ip() should return False for public IPv4 "
            f"'{public_ip}'"
        )

    @settings(max_examples=100)
    @given(private_ip=ipv4_10_range())
    def test_property_3_10_range_blocked(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IP in 10.0.0.0/8 range, _is_private_ip() SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        assert protector._is_private_ip(private_ip) is True, (
            f"10.x.x.x range should be blocked: {private_ip}"
        )

    @settings(max_examples=100)
    @given(private_ip=ipv4_172_16_range())
    def test_property_3_172_16_range_blocked(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IP in 172.16.0.0/12 range, _is_private_ip() SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        assert protector._is_private_ip(private_ip) is True, (
            f"172.16-31.x.x range should be blocked: {private_ip}"
        )

    @settings(max_examples=100)
    @given(private_ip=ipv4_192_168_range())
    def test_property_3_192_168_range_blocked(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IP in 192.168.0.0/16 range, _is_private_ip() SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        assert protector._is_private_ip(private_ip) is True, (
            f"192.168.x.x range should be blocked: {private_ip}"
        )

    @settings(max_examples=100)
    @given(private_ip=ipv4_127_range())
    def test_property_3_127_range_blocked(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IP in 127.0.0.0/8 range (loopback), _is_private_ip()
        SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        assert protector._is_private_ip(private_ip) is True, (
            f"127.x.x.x loopback range should be blocked: {private_ip}"
        )

    def test_property_3_ipv6_loopback_blocked(self):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        The IPv6 loopback address (::1) SHALL be blocked.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        assert protector._is_private_ip("::1") is True, (
            "IPv6 loopback ::1 should be blocked"
        )

    def test_property_3_invalid_ip_treated_as_private(self):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        Invalid IP address formats SHALL be treated as private (fail-safe).

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        # Invalid IP formats should be treated as private for safety
        invalid_ips = [
            "not-an-ip",
            "256.256.256.256",
            "1.2.3.4.5",
            "",
            "abc::xyz",
        ]

        for invalid_ip in invalid_ips:
            assert protector._is_private_ip(invalid_ip) is True, (
                f"Invalid IP '{invalid_ip}' should be treated as private"
            )

    @settings(max_examples=50)
    @given(private_ip=ipv4_169_254_range())
    def test_property_3_link_local_ipv4_blocked(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IP in 169.254.0.0/16 (link-local), _is_private_ip()
        SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        assert protector._is_private_ip(private_ip) is True, (
            f"169.254.x.x link-local range should be blocked: {private_ip}"
        )

    @settings(max_examples=50)
    @given(private_ip=ipv4_multicast_range())
    def test_property_3_multicast_ipv4_blocked(self, private_ip):
        """
        **Feature: agent-security-hardening, Property 3: Private IP Blocking**

        For any IP in 224.0.0.0/4 (multicast), _is_private_ip()
        SHALL return True.

        **Validates: Requirements 2.9**
        """
        from core.security.ssrf_protector import SSRFProtector, DomainAllowlist

        allowlist = DomainAllowlist(["example.com"])
        protector = SSRFProtector(allowlist, block_private_ips=True)

        assert protector._is_private_ip(private_ip) is True, (
            f"224-239.x.x.x multicast range should be blocked: {private_ip}"
        )
