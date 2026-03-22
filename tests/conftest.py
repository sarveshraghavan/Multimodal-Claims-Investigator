"""
Pytest configuration and shared fixtures.
"""
import pytest
import os
from pathlib import Path

# Set test environment variables
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "test_api_key")

@pytest.fixture
def test_uploads_dir(tmp_path):
    """Create a temporary uploads directory for testing."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    return uploads

@pytest.fixture
def sample_claim_id():
    """Provide a sample claim ID for testing."""
    return "claim-test-12345"
