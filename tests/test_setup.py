"""
Basic tests to verify project setup.
"""

import pytest


def test_imports():
    """Test that core modules can be imported"""
    import api
    import core
    import collectors
    import models
    import config
    
    assert api is not None
    assert core is not None
    assert collectors is not None
    assert models is not None
    assert config is not None


def test_main_module():
    """Test that main module can be imported"""
    import main
    
    assert main.app is not None
    assert hasattr(main.app, 'get')
