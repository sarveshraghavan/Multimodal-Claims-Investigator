"""
Tests for main FastAPI application.
"""
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root health check endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "multimodal-claims-investigator"

@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the detailed health check endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gemini_api_configured" in data

@pytest.mark.asyncio
async def test_ingest_endpoint_success():
    """Test successful file ingestion via POST /ingest endpoint."""
    from unittest.mock import patch, AsyncMock
    from io import BytesIO
    
    # Mock the ingest_file function
    mock_result = {
        "file_id": "test-uuid-123",
        "filename": "test_video.mp4",
        "modality": "video",
        "claim_id": "claim-456",
        "status": "success"
    }
    
    with patch('app.main.ingest_file', new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Create multipart form data
            files = {"file": ("test_video.mp4", BytesIO(b"test video content"), "video/mp4")}
            data = {"claim_id": "claim-456", "description": "Test video"}
            
            response = await client.post("/ingest", files=files, data=data)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["file_id"] == "test-uuid-123"
        assert result["filename"] == "test_video.mp4"
        assert result["modality"] == "video"
        assert result["claim_id"] == "claim-456"
        assert result["status"] == "success"
        
        # Verify ingest_file was called
        mock_ingest.assert_called_once()


@pytest.mark.asyncio
async def test_ingest_endpoint_validation_error():
    """Test that validation errors return 400 status code with consistent format."""
    from unittest.mock import patch, AsyncMock
    from io import BytesIO
    
    # Mock ingest_file to raise ValueError (validation error)
    with patch('app.main.ingest_file', new_callable=AsyncMock) as mock_ingest:
        mock_ingest.side_effect = ValueError("Unsupported file type. Supported types: .mp4, .jpg, .mp3, .pdf")
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("document.txt", BytesIO(b"text content"), "text/plain")}
            data = {"claim_id": "", "description": ""}
            
            response = await client.post("/ingest", files=files, data=data)
        
        # Verify 400 status code and consistent error format
        assert response.status_code == 400
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
        assert "Unsupported file type" in result["error"]


@pytest.mark.asyncio
async def test_ingest_endpoint_internal_error():
    """Test that internal errors return 500 status code with consistent format."""
    from unittest.mock import patch, AsyncMock
    from io import BytesIO
    
    # Mock ingest_file to raise generic Exception (internal error)
    with patch('app.main.ingest_file', new_callable=AsyncMock) as mock_ingest:
        mock_ingest.side_effect = Exception("Database connection failed")
        
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            files = {"file": ("test_video.mp4", BytesIO(b"video content"), "video/mp4")}
            data = {"claim_id": "claim-789", "description": "Test"}
            
            response = await client.post("/ingest", files=files, data=data)
        
        # Verify 500 status code and consistent error format
        assert response.status_code == 500
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "InternalError"
        assert "Internal server error" in result["error"]


@pytest.mark.asyncio
async def test_ingest_endpoint_optional_parameters():
    """Test that claim_id and description are optional."""
    from unittest.mock import patch, AsyncMock
    from io import BytesIO
    
    mock_result = {
        "file_id": "test-uuid-456",
        "filename": "test_image.jpg",
        "modality": "image",
        "claim_id": "",
        "status": "success"
    }
    
    with patch('app.main.ingest_file', new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Send only file, no claim_id or description
            files = {"file": ("test_image.jpg", BytesIO(b"image content"), "image/jpeg")}
            
            response = await client.post("/ingest", files=files)
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        
        # Verify ingest_file was called with empty strings for optional params
        mock_ingest.assert_called_once()
        call_args = mock_ingest.call_args
        assert call_args[0][1] == ""  # claim_id
        assert call_args[0][2] == ""  # description


@pytest.mark.asyncio
async def test_ingest_endpoint_multipart_form_data():
    """Test that endpoint correctly accepts multipart form data."""
    from unittest.mock import patch, AsyncMock
    from io import BytesIO
    
    mock_result = {
        "file_id": "test-uuid-789",
        "filename": "test_audio.mp3",
        "modality": "audio",
        "claim_id": "claim-999",
        "status": "success"
    }
    
    with patch('app.main.ingest_file', new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("test_audio.mp3", BytesIO(b"audio content"), "audio/mpeg")}
            data = {
                "claim_id": "claim-999",
                "description": "Audio recording from incident"
            }
            
            response = await client.post("/ingest", files=files, data=data)
        
        assert response.status_code == 200
        
        # Verify the correct parameters were passed to ingest_file
        call_args = mock_ingest.call_args
        # First argument is the UploadFile object
        assert call_args[0][0].filename == "test_audio.mp3"
        # Second argument is claim_id
        assert call_args[0][1] == "claim-999"
        # Third argument is description
        assert call_args[0][2] == "Audio recording from incident"


@pytest.mark.asyncio
async def test_search_endpoint_success():
    """Test successful search via POST /search endpoint."""
    from unittest.mock import patch, AsyncMock
    
    # Mock the search function
    mock_result = {
        "results": [
            {
                "file_id": "file-1",
                "filename": "video1.mp4",
                "modality": "video",
                "claim_id": "claim-123",
                "similarity": 0.95,
                "metadata": {
                    "filename": "video1.mp4",
                    "modality": "video",
                    "claim_id": "claim-123",
                    "path": "/uploads/claim-123/video/video1.mp4",
                    "file_size": 1024,
                    "upload_timestamp": "2024-01-01T00:00:00Z"
                }
            }
        ],
        "query": "dashcam footage",
        "total_results": 1
    }
    
    with patch('app.main.search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={
                    "query": "dashcam footage",
                    "top_k": 10,
                    "claim_id": "claim-123"
                }
            )
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["query"] == "dashcam footage"
        assert result["total_results"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["filename"] == "video1.mp4"
        assert result["results"][0]["similarity"] == 0.95
        
        # Verify search was called with correct parameters
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "dashcam footage"
        assert call_kwargs["top_k"] == 10
        assert call_kwargs["claim_id"] == "claim-123"


@pytest.mark.asyncio
async def test_search_endpoint_default_parameters():
    """Test search endpoint with default parameters."""
    from unittest.mock import patch, AsyncMock
    
    mock_result = {
        "results": [],
        "query": "test query",
        "total_results": 0
    }
    
    with patch('app.main.search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={"query": "test query"}
            )
        
        # Verify response
        assert response.status_code == 200
        
        # Verify search was called with default parameters
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["query"] == "test query"
        assert call_kwargs["top_k"] == 10  # Default value
        assert call_kwargs["claim_id"] == ""  # Default value


@pytest.mark.asyncio
async def test_search_endpoint_validation_error():
    """Test that validation errors return 400 status code with consistent format."""
    from unittest.mock import patch, AsyncMock
    
    # Mock search to raise ValueError (validation error)
    with patch('app.main.search', new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = ValueError("Query cannot be empty")
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={"query": "", "top_k": 10}
            )
        
        # Verify 400 status code and consistent error format
        assert response.status_code == 400
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
        assert "Query cannot be empty" in result["error"]


@pytest.mark.asyncio
async def test_search_endpoint_internal_error():
    """Test that internal errors return 500 status code with consistent format."""
    from unittest.mock import patch, AsyncMock
    
    # Mock search to raise generic Exception (internal error)
    with patch('app.main.search', new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = Exception("Embedding service unavailable")
        
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={"query": "test query"}
            )
        
        # Verify 500 status code and consistent error format
        assert response.status_code == 500
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "InternalError"
        assert "Internal server error" in result["error"]


@pytest.mark.asyncio
async def test_search_endpoint_empty_results():
    """Test search endpoint with no matching results."""
    from unittest.mock import patch, AsyncMock
    
    mock_result = {
        "results": [],
        "query": "nonexistent content",
        "total_results": 0
    }
    
    with patch('app.main.search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={"query": "nonexistent content", "top_k": 5}
            )
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["total_results"] == 0
        assert len(result["results"]) == 0
        assert result["query"] == "nonexistent content"


@pytest.mark.asyncio
async def test_search_endpoint_claim_filtering():
    """Test search endpoint with claim_id filtering."""
    from unittest.mock import patch, AsyncMock
    
    mock_result = {
        "results": [
            {
                "file_id": "file-1",
                "filename": "video1.mp4",
                "modality": "video",
                "claim_id": "claim-456",
                "similarity": 0.88,
                "metadata": {}
            }
        ],
        "query": "accident footage",
        "total_results": 1
    }
    
    with patch('app.main.search', new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={
                    "query": "accident footage",
                    "top_k": 20,
                    "claim_id": "claim-456"
                }
            )
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert result["results"][0]["claim_id"] == "claim-456"
        
        # Verify search was called with claim_id filter
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["claim_id"] == "claim-456"


@pytest.mark.asyncio
async def test_investigate_endpoint_success():
    """Test successful investigation via POST /investigate endpoint."""
    from unittest.mock import patch, AsyncMock
    
    # Mock the investigate function
    mock_result = {
        "answer": "Based on the dashcam video, the collision occurred at 2:30 PM.",
        "sources": [
            {
                "filename": "dashcam.mp4",
                "modality": "video",
                "claim_id": "claim-123",
                "similarity": 0.92
            }
        ],
        "model_used": "gemini-2.5-pro-preview-05-06"
    }
    
    with patch('app.main.investigate', new_callable=AsyncMock) as mock_investigate:
        mock_investigate.return_value = mock_result
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/investigate",
                json={
                    "question": "When did the collision occur?",
                    "claim_id": "claim-123",
                    "top_k": 6
                }
            )
        
        # Verify response
        assert response.status_code == 200
        result = response.json()
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["filename"] == "dashcam.mp4"
        
        # Verify investigate was called with correct parameters
        mock_investigate.assert_called_once()
        call_kwargs = mock_investigate.call_args[1]
        assert call_kwargs["question"] == "When did the collision occur?"
        assert call_kwargs["claim_id"] == "claim-123"
        assert call_kwargs["top_k"] == 6


@pytest.mark.asyncio
async def test_investigate_endpoint_validation_error():
    """Test that validation errors return 400 status code with consistent format."""
    from unittest.mock import patch, AsyncMock
    
    # Mock investigate to raise ValueError (validation error)
    with patch('app.main.investigate', new_callable=AsyncMock) as mock_investigate:
        mock_investigate.side_effect = ValueError("Question cannot be empty")
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/investigate",
                json={"question": "", "claim_id": "claim-123"}
            )
        
        # Verify 400 status code and consistent error format
        assert response.status_code == 400
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
        assert "Question cannot be empty" in result["error"]


@pytest.mark.asyncio
async def test_investigate_endpoint_internal_error():
    """Test that internal errors return 500 status code with consistent format."""
    from unittest.mock import patch, AsyncMock
    
    # Mock investigate to raise generic Exception (internal error)
    with patch('app.main.investigate', new_callable=AsyncMock) as mock_investigate:
        mock_investigate.side_effect = Exception("LLM service unavailable")
        
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            response = await client.post(
                "/investigate",
                json={"question": "What happened?"}
            )
        
        # Verify 500 status code and consistent error format
        assert response.status_code == 500
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "InternalError"
        assert "Internal server error" in result["error"]


@pytest.mark.asyncio
async def test_global_exception_handler_request_validation():
    """Test global exception handler for Pydantic validation errors."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Send invalid JSON to search endpoint (missing required field)
        response = await client.post(
            "/search",
            json={"top_k": "invalid"}  # top_k should be int, not string
        )
    
    # Verify 400 status code and consistent error format
    assert response.status_code == 400
    result = response.json()
    assert "error" in result
    assert "error_type" in result
    assert result["error_type"] == "ValidationError"
    assert "details" in result


@pytest.mark.asyncio
async def test_global_exception_handler_not_found():
    """Test global exception handler for NotFoundException."""
    from unittest.mock import patch, AsyncMock
    from app.main import NotFoundException
    
    # Mock search to raise NotFoundException
    with patch('app.main.search', new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = NotFoundException("Resource not found")
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/search",
                json={"query": "test"}
            )
        
        # Verify 404 status code and consistent error format
        assert response.status_code == 404
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "NotFoundError"
        assert "Resource not found" in result["error"]


@pytest.mark.asyncio
async def test_consistent_error_response_format():
    """Test that all error responses follow consistent format."""
    from unittest.mock import patch, AsyncMock
    
    # Test ValueError (400)
    with patch('app.main.ingest_file', new_callable=AsyncMock) as mock_ingest:
        mock_ingest.side_effect = ValueError("Test validation error")
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            from io import BytesIO
            files = {"file": ("test.mp4", BytesIO(b"content"), "video/mp4")}
            response = await client.post("/ingest", files=files)
        
        assert response.status_code == 400
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "ValidationError"
    
    # Test generic Exception (500)
    with patch('app.main.ingest_file', new_callable=AsyncMock) as mock_ingest:
        mock_ingest.side_effect = Exception("Test internal error")
        
        async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as client:
            from io import BytesIO
            files = {"file": ("test.mp4", BytesIO(b"content"), "video/mp4")}
            response = await client.post("/ingest", files=files)
        
        assert response.status_code == 500
        result = response.json()
        assert "error" in result
        assert "error_type" in result
        assert result["error_type"] == "InternalError"
