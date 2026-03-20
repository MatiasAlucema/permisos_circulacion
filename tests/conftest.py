"""Shared fixtures for all tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest

import api.main as api_module
from src.monitoring import get_monitor
from src.predict import get_predictor


@pytest.fixture(scope="session", autouse=True)
def setup_api():
    """Initialize model and monitor before API tests (replaces lifespan)."""
    api_module.predictor = get_predictor()
    api_module.monitor = get_monitor()
