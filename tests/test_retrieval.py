"""
Tests for the investigation service (retrieval.py).
"""
import pytest
from unittest.mock import Mock, patch, mock_open
from app.retrieval import _mime_for_modality, investigate


class TestMimeForModality:
    """Tests for MIME type mapping function."""
    
    def test_video_mime_types(self):
        """Test MIME type mapping for video files."""
        assert _mime_for_modality("video", ".mp4") == "video/mp4"
        assert _mime_for_modality("video", ".mov") == "video/quicktime"
        assert _mime_for_modality("video", ".avi") == "video/x-msvideo"
        assert _mime_for_modality("video", ".mkv") == "video/x-matroska"
        assert _mime_for_modality("video", ".webm") == "video/webm"
    
    def test_image_mime_types(self):
        """Test MIME type mapping for image files."""
        assert _mime_for_modality("image", ".jpg") == "image/jpeg"
        assert _mime_for_modality("image", ".jpeg") == "image/jpeg"
        assert _mime_for_modality("image", ".png") == "image/png"
        assert _mime_for_modality("image", ".gif") == "image/gif"
        assert _mime_for_modality("image", ".webp") == "image/webp"
    
    def test_audio_mime_types(self):
        """Test MIME type mapping for audio files."""
        assert _mime_for_modality("audio", ".mp3") == "audio/mpeg"
        assert _mime_for_modality("audio", ".wav") == "audio/wav"
        assert _mime_for_modality("audio", ".m4a") == "audio/mp4"
        assert _mime_for_modality("audio", ".flac") == "audio/flac"
        assert _mime_for_modality("audio", ".ogg") == "audio/ogg"
    
    def test_document_mime_types(self):
        """Test MIME type mapping for document files."""
        assert _mime_for_modality("document", ".pdf") == "application/pdf"
    
    def test_case_insensitive(self):
        """Test that MIME type mapping is case-insensitive."""
        assert _mime_for_modality("video", ".MP4") == "video/mp4"
        assert _mime_for_modality("image", ".JPG") == "image/jpeg"
    
    def test_unknown_extension_fallback(self):
        """Test fallback for unknown extensions."""
        assert _mime_for_modality("video", ".unknown") == "video/mp4"
        assert _mime_for_modality("image", ".unknown") == "image/jpeg"
        assert _mime_for_modality("audio", ".unknown") == "audio/mpeg"


class TestInvestigate:
    """Tests for the main investigation handler."""
    
    @pytest.mark.asyncio
    async def test_empty_question_raises_error(self):
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await investigate("")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await investigate("   ")
    
    @pytest.mark.asyncio
    async def test_invalid_top_k_raises_error(self):
        """Test that invalid top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            await investigate("test question", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            await investigate("test question", top_k=-1)
    
    @pytest.mark.asyncio
    @patch('app.retrieval.embed_text')
    @patch('app.retrieval.search_embeddings')
    async def test_no_evidence_found(self, mock_search, mock_embed):
        """Test handling when no evidence files are found."""
        # Mock empty search results
        mock_embed.return_value = [0.1] * 768
        mock_search.return_value = {
            "metadatas": [],
            "distances": []
        }
        
        result = await investigate("test question")
        
        assert "No evidence files found" in result["answer"]
        assert result["sources"] == []
        assert "model_used" in result
    
    @pytest.mark.asyncio
    @patch('app.retrieval.embed_text')
    @patch('app.retrieval.search_embeddings')
    @patch('app.retrieval._call_gemini')
    async def test_successful_investigation(self, mock_gemini, mock_search, mock_embed):
        """Test successful investigation with evidence."""
        # Mock embedding
        mock_embed.return_value = [0.1] * 768
        
        # Mock search results
        mock_search.return_value = {
            "metadatas": [
                {
                    "filename": "test.jpg",
                    "modality": "image",
                    "claim_id": "claim-123",
                    "path": "/path/to/test.jpg"
                }
            ],
            "distances": [0.5]
        }
        
        # Mock Gemini response
        mock_gemini.return_value = (
            "Analysis complete: The image shows damage to the vehicle.",
            [{"filename": "test.jpg", "modality": "image", "claim_id": "claim-123", "similarity": 0.67}]
        )
        
        result = await investigate("What damage is visible?", claim_id="claim-123", top_k=5)
        
        assert "Analysis complete" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "test.jpg"
        assert "model_used" in result
        
        # Verify the mocks were called correctly
        mock_embed.assert_called_once()
        mock_search.assert_called_once_with(
            query_embedding=[0.1] * 768,
            top_k=5,
            claim_id="claim-123"
        )
