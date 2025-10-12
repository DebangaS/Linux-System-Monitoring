import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up global test environment variables"""
    # Speed mode for process operations
    os.environ.setdefault("FAST_MODE", "1")
    # Ensure background tasks are disabled
    os.environ.setdefault("APP_ENABLE_BACKGROUND", "0")
    
    yield
    
    # Cleanup is handled by individual test classes

