"""
Tests for the /investigate endpoint in main.py.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)


class TestInvestigateEndpoint:
    """Tests for the POST /investigate endpoint."""
    
    @patch('app.main.investigate')
    def test_successful_investigation(self, mock_investigate):
        """Test successful investigation request."""
        # Mock the investigate function
        mock_investigate.return_value = {
            "answer": "The dashcam shows a collision at the intersection.",
            "sources": [
                {
                    "filename": "dashcam.mp4",
                    "modality": "video",
                    "claim_id": "claim-123",
                    "similarity": 0.92
                }
            ],
            "model_used": "gemini-2.0-flash-exp"
        }
        
        # Make request
        response = client.post(
            "/investigate",
            json={
                "question": "What does the dashcam show?",
                "claim_id": "claim-123",
                "top_k": 5
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "model_used" in data
        assert data["answer"] == "The dashcam shows a collision at the intersection."
        assert len(data["sources"]) == 1
    
    @patch('app.main.investigate')
    def test_investigation_with_defaults(self, mock_investigate):
        """Test investigation with default parameters."""
        mock_investigate.return_value = {
            "answer": "Analysis complete.",
            "sources": [],
            "model_used": "gemini-2.0-flash-exp"
        }
        
        # Make request with minimal payload
        response = client.post(
            "/investigate",
            json={"question": "What happened?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    @patch('app.main.investigate')
    def test_validation_error_returns_400(self, mock_investigate):
        """Test that validation errors return 400 status."""
        # Mock investigate to raise ValueError
        mock_investigate.side_effect = ValueError("Question cannot be empty")
        
        response = client.post(
            "/investigate",
            json={"question": ""}
        )
        
        assert response.status_code == 400
        assert "Question cannot be empty" in response.json()["detail"]
    
    @patch('app.main.investigate')
    def test_internal_error_returns_500(self, mock_investigate):
        """Test that internal errors return 500 status."""
        # Mock investigate to raise generic exception
        mock_investigate.side_effect = Exception("Database connection failed")
        
        response = client.post(
            "/investigate",
            json={"question": "What happened?"}
        )
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]
    
    def test_malformed_json_returns_422(self):
        """Test that malformed JSON returns 422 status."""
        response = client.post(
            "/investigate",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_field_returns_422(self):
        """Test that missing required field returns 422 status."""
        response = client.post(
            "/investigate",
            json={"claim_id": "claim-123"}  # Missing required 'question' field
        )
        
        assert response.status_code == 422
