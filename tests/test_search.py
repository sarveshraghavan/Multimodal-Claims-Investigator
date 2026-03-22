"""
Tests for search service functionality.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.search import calculate_similarity, search


class TestCalculateSimilarity:
    """Tests for similarity score calculation."""
    
    def test_zero_distance_returns_one(self):
        """Distance of 0 should return similarity of 1.0 (identical)."""
        assert calculate_similarity(0.0) == 1.0
    
    def test_distance_one_returns_half(self):
        """Distance of 1 should return similarity of 0.5."""
        assert calculate_similarity(1.0) == 0.5
    
    def test_small_distance_high_similarity(self):
        """Small distances should return high similarity scores."""
        similarity = calculate_similarity(0.1)
        assert similarity > 0.9
        assert similarity < 1.0
    
    def test_large_distance_low_similarity(self):
        """Large distances should return low similarity scores."""
        similarity = calculate_similarity(10.0)
        assert similarity < 0.1
        assert similarity > 0.0
    
    def test_similarity_always_positive(self):
        """Similarity scores should always be positive."""
        for distance in [0, 0.5, 1, 5, 10, 100]:
            assert calculate_similarity(distance) > 0


class TestSearch:
    """Tests for main search handler."""
    
    @pytest.mark.asyncio
    async def test_search_empty_query_raises_error(self):
        """Empty query should raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search("")
    
    @pytest.mark.asyncio
    async def test_search_whitespace_query_raises_error(self):
        """Whitespace-only query should raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search("   ")
    
    @pytest.mark.asyncio
    async def test_search_invalid_top_k_raises_error(self):
        """Invalid top_k values should raise ValueError."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            await search("test query", top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            await search("test query", top_k=-1)
    
    @pytest.mark.asyncio
    @patch('app.search.embed_text')
    @patch('app.search.search_embeddings')
    async def test_search_returns_formatted_results(self, mock_search_embeddings, mock_embed_text):
        """Search should return properly formatted results with similarity scores."""
        # Mock embedding
        mock_embed_text.return_value = [0.1] * 768
        
        # Mock search results
        mock_search_embeddings.return_value = {
            "ids": ["file-1", "file-2"],
            "metadatas": [
                {
                    "filename": "video1.mp4",
                    "modality": "video",
                    "claim_id": "claim-123",
                    "path": "/uploads/claim-123/video/video1.mp4",
                    "file_size": 1024,
                    "upload_timestamp": "2024-01-01T00:00:00Z"
                },
                {
                    "filename": "image1.jpg",
                    "modality": "image",
                    "claim_id": "claim-123",
                    "path": "/uploads/claim-123/image/image1.jpg",
                    "file_size": 512,
                    "upload_timestamp": "2024-01-01T00:00:00Z"
                }
            ],
            "distances": [0.1, 0.5]
        }
        
        # Execute search
        result = await search("test query", top_k=10, claim_id="claim-123")
        
        # Verify structure
        assert "results" in result
        assert "query" in result
        assert "total_results" in result
        
        # Verify query echoed back
        assert result["query"] == "test query"
        
        # Verify result count
        assert result["total_results"] == 2
        assert len(result["results"]) == 2
        
        # Verify first result
        first_result = result["results"][0]
        assert first_result["file_id"] == "file-1"
        assert first_result["filename"] == "video1.mp4"
        assert first_result["modality"] == "video"
        assert first_result["claim_id"] == "claim-123"
        assert "similarity" in first_result
        assert first_result["similarity"] > 0
        assert "metadata" in first_result
        
        # Verify similarity scores are calculated correctly
        # Distance 0.1 should give similarity ~0.909
        assert first_result["similarity"] > 0.9
        
        # Distance 0.5 should give similarity ~0.667
        second_result = result["results"][1]
        assert 0.6 < second_result["similarity"] < 0.7
    
    @pytest.mark.asyncio
    @patch('app.search.embed_text')
    @patch('app.search.search_embeddings')
    async def test_search_with_no_results(self, mock_search_embeddings, mock_embed_text):
        """Search with no results should return empty results list."""
        # Mock embedding
        mock_embed_text.return_value = [0.1] * 768
        
        # Mock empty search results
        mock_search_embeddings.return_value = {
            "ids": [],
            "metadatas": [],
            "distances": []
        }
        
        # Execute search
        result = await search("test query")
        
        # Verify empty results
        assert result["total_results"] == 0
        assert len(result["results"]) == 0
        assert result["query"] == "test query"
    
    @pytest.mark.asyncio
    @patch('app.search.embed_text')
    @patch('app.search.search_embeddings')
    async def test_search_passes_parameters_correctly(self, mock_search_embeddings, mock_embed_text):
        """Search should pass parameters correctly to underlying functions."""
        # Mock embedding
        mock_embed_text.return_value = [0.1] * 768
        
        # Mock search results
        mock_search_embeddings.return_value = {
            "ids": [],
            "metadatas": [],
            "distances": []
        }
        
        # Execute search with specific parameters
        await search("test query", top_k=5, claim_id="claim-456")
        
        # Verify embed_text was called with correct parameters
        mock_embed_text.assert_called_once()
        call_args = mock_embed_text.call_args
        assert call_args[0][0] == "test query"
        
        # Verify search_embeddings was called with correct parameters
        mock_search_embeddings.assert_called_once()
        call_kwargs = mock_search_embeddings.call_args[1]
        assert call_kwargs["top_k"] == 5
        assert call_kwargs["claim_id"] == "claim-456"
    
    @pytest.mark.asyncio
    @patch('app.search.embed_text')
    @patch('app.search.search_embeddings')
    async def test_search_rounds_similarity_scores(self, mock_search_embeddings, mock_embed_text):
        """Similarity scores should be rounded to 4 decimal places."""
        # Mock embedding
        mock_embed_text.return_value = [0.1] * 768
        
        # Mock search results with distance that produces many decimals
        mock_search_embeddings.return_value = {
            "ids": ["file-1"],
            "metadatas": [{
                "filename": "test.jpg",
                "modality": "image",
                "claim_id": "",
                "path": "/test.jpg",
                "file_size": 100,
                "upload_timestamp": "2024-01-01T00:00:00Z"
            }],
            "distances": [0.123456789]
        }
        
        # Execute search
        result = await search("test query")
        
        # Verify similarity is rounded to 4 decimal places
        similarity = result["results"][0]["similarity"]
        # Convert to string to check decimal places
        similarity_str = str(similarity)
        decimal_places = len(similarity_str.split('.')[-1]) if '.' in similarity_str else 0
        assert decimal_places <= 4
