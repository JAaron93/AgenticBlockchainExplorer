import pytest
import asyncio
from unittest.mock import MagicMock, patch
import logging

# Import the module under test
import collectors.base as base

@pytest.mark.asyncio
async def test_init_collector_locks_creates_locks():
    """Verify that init_collector_locks correctly initializes the locks."""
    # Reset module state
    base._secure_http_client_lock = None
    base._schema_validator_lock = None
    
    # Call init function synchronously
    base.init_collector_locks()
    
    # Verify locks are created
    assert isinstance(base._secure_http_client_lock, asyncio.Lock)
    assert isinstance(base._schema_validator_lock, asyncio.Lock)
    
    # Verify they work
    async with base._secure_http_client_lock:
        pass
        
    async with base._schema_validator_lock:
        pass

@pytest.mark.asyncio
async def test_init_collector_locks_idempotency():
    """Verify that calling init multiple times is safe."""
    # Reset
    base._secure_http_client_lock = None
    base._schema_validator_lock = None
    
    # First call synchronously
    base.init_collector_locks()
    lock1 = base._secure_http_client_lock
    lock2 = base._schema_validator_lock
    
    # Second call synchronously
    base.init_collector_locks()
    
    # Should be same instances (or at least valid and not overwritten if checks work)
    # Our implementation checks 'if is None', so it should preserve existing locks
    assert base._secure_http_client_lock is lock1
    assert base._schema_validator_lock is lock2
