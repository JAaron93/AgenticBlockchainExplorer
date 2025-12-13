"""Security components for the blockchain stablecoin explorer.

This package provides security hardening components:

Currently implemented:
- CredentialSanitizer: Detects and redacts credentials from strings/dicts/URLs
- SecureLogger: Logger wrapper with automatic credential sanitization
- DomainAllowlist: Validates domains against configurable allowlist
- SSRFProtector: Protects against SSRF attacks with IP blocking

Additional components will be added as the security hardening spec progresses:
- Resource exhaustion protection
- Blockchain data validation
- Safe file path handling
"""

from core.security.credential_sanitizer import CredentialSanitizer
from core.security.secure_logger import SecureLogger
from core.security.ssrf_protector import (
    DomainAllowlist,
    DomainNotAllowedError,
    DNSRebindingError,
    DNSResolutionError,
    PrivateIPError,
    ProtocolNotAllowedError,
    SSRFError,
    SSRFProtector,
)

__all__ = [
    "CredentialSanitizer",
    "SecureLogger",
    "DomainAllowlist",
    "DomainNotAllowedError",
    "DNSRebindingError",
    "DNSResolutionError",
    "PrivateIPError",
    "ProtocolNotAllowedError",
    "SSRFError",
    "SSRFProtector",
]
