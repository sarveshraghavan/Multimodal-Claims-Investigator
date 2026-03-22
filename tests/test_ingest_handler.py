"""Unit tests for the main ingest_file handler function."""
import pytest
from fastapi import UploadFile
from io import BytesIO
import os
import shutil
from unittest.mock import patch, AsyncMock


def create_upload_file(filename: str, content: bytes = b"test", size: int = None):
    """Helper to create mock UploadFile."""
    file = UploadFile(filename=filename, file=BytesIO(content))
    file.size = size if size is not None else len(content)
    return file


class TestIngestFile:
    """Tests for the main ingest_file function."""
    
    @pytest.mark.asyncio
    async def test_ingest_file_success(self):
        """Test successful file ingestion returns correct response."""
        from app.ingest import ingest_file
        
        test_dir = "uploads/test-claim-ingest"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            # Create test file
            file = create_upload_file("test_video.mp4", b"test video content", 1024)
            
            # Mock the embedding and database functions
            mock_embedding = [0.1] * 768  # Mock embedding vector
            
            with patch('app.ingest.embed_file', new_callable=AsyncMock) as mock_embed:
                with patch('app.ingest.store_embedding') as mock_store:
                    mock_embed.return_value = mock_embedding
                    mock_store.return_value = "test-file-id"
                    
                    # Call ingest_file
                    result = await ingest_file(file, claim_id="test-claim-ingest", description="Test video")
                    
                    # Verify response structure
                    assert "file_id" in result
                    assert result["filename"] == "test_video.mp4"
                    assert result["modality"] == "video"
                    assert result["claim_id"] == "test-claim-ingest"
                    assert result["status"] == "success"
                    
                    # Verify embed_file was called with correct arguments
                    mock_embed.assert_called_once()
                    call_args = mock_embed.call_args
                    assert "test_video.mp4" in call_args[0][0]  # file_path contains filename
                    assert call_args[0][1] == "video"  # modality
                    
                    # Verify store_embedding was called
                    mock_store.assert_called_once()
                    store_args = mock_store.call_args
                    assert store_args[0][0] == mock_embedding  # embedding
                    assert store_args[0][1]["filename"] == "test_video.mp4"  # metadata
                    assert store_args[0][1]["modality"] == "video"
                    assert store_args[0][1]["claim_id"] == "test-claim-ingest"
                    assert store_args[0][1]["description"] == "Test video"
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_ingest_file_validation_failure(self):
        """Test that validation failure raises ValueError."""
        from app.ingest import ingest_file
        
        # Create invalid file (unsupported type)
        file = create_upload_file("document.txt", b"text content", 1024)
        
        # Should raise ValueError due to validation failure
        with pytest.raises(ValueError, match="Unsupported file type"):
            await ingest_file(file, claim_id="test-claim")
    
    @pytest.mark.asyncio
    async def test_ingest_file_cleanup_on_embedding_failure(self):
        """Test that file is cleaned up if embedding generation fails."""
        from app.ingest import ingest_file
        
        test_dir = "uploads/test-claim-cleanup"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            file = create_upload_file("test_image.jpg", b"test image content", 2048)
            
            # Mock embed_file to raise an exception
            with patch('app.ingest.embed_file', new_callable=AsyncMock) as mock_embed:
                mock_embed.side_effect = Exception("Embedding API failed")
                
                # Should raise exception
                with pytest.raises(Exception, match="Embedding API failed"):
                    await ingest_file(file, claim_id="test-claim-cleanup")
                
                # Verify file was cleaned up (check both possible path formats)
                possible_paths = [
                    "uploads/test-claim-cleanup/image/test_image.jpg",
                    "uploads\\test-claim-cleanup\\image\\test_image.jpg"
                ]
                for expected_path in possible_paths:
                    if os.path.exists(expected_path):
                        assert False, f"File should be cleaned up after failure but found at {expected_path}"
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_ingest_file_cleanup_on_storage_failure(self):
        """Test that file is cleaned up if database storage fails."""
        from app.ingest import ingest_file
        
        test_dir = "uploads/test-claim-storage-fail"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            file = create_upload_file("test_audio.mp3", b"test audio content", 3072)
            
            mock_embedding = [0.2] * 768
            
            # Mock embed_file to succeed but store_embedding to fail
            with patch('app.ingest.embed_file', new_callable=AsyncMock) as mock_embed:
                with patch('app.ingest.store_embedding') as mock_store:
                    mock_embed.return_value = mock_embedding
                    mock_store.side_effect = Exception("Database storage failed")
                    
                    # Should raise exception
                    with pytest.raises(Exception, match="Database storage failed"):
                        await ingest_file(file, claim_id="test-claim-storage-fail")
                    
                    # Verify file was cleaned up (check both possible path formats)
                    possible_paths = [
                        "uploads/test-claim-storage-fail/audio/test_audio.mp3",
                        "uploads\\test-claim-storage-fail\\audio\\test_audio.mp3"
                    ]
                    for expected_path in possible_paths:
                        if os.path.exists(expected_path):
                            assert False, f"File should be cleaned up after storage failure but found at {expected_path}"
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_ingest_file_with_empty_claim_id(self):
        """Test ingestion with empty claim_id uses 'unclaimed' directory."""
        from app.ingest import ingest_file
        
        test_dir = "uploads/unclaimed"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            file = create_upload_file("test_doc.pdf", b"test pdf content", 4096)
            
            mock_embedding = [0.3] * 768
            
            with patch('app.ingest.embed_file', new_callable=AsyncMock) as mock_embed:
                with patch('app.ingest.store_embedding') as mock_store:
                    mock_embed.return_value = mock_embedding
                    mock_store.return_value = "test-file-id"
                    
                    # Call with empty claim_id
                    result = await ingest_file(file, claim_id="", description="")
                    
                    # Verify response
                    assert result["claim_id"] == ""
                    assert result["status"] == "success"
                    
                    # Verify metadata passed to store_embedding has empty claim_id
                    store_args = mock_store.call_args
                    assert store_args[0][1]["claim_id"] == ""
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_ingest_file_metadata_completeness(self):
        """Test that all required metadata fields are included."""
        from app.ingest import ingest_file
        
        test_dir = "uploads/test-claim-metadata"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            file = create_upload_file("test_video.mp4", b"test content", 5120)
            
            mock_embedding = [0.4] * 768
            
            with patch('app.ingest.embed_file', new_callable=AsyncMock) as mock_embed:
                with patch('app.ingest.store_embedding') as mock_store:
                    mock_embed.return_value = mock_embedding
                    mock_store.return_value = "test-file-id"
                    
                    await ingest_file(file, claim_id="test-claim-metadata", description="Test description")
                    
                    # Verify metadata has all required fields
                    store_args = mock_store.call_args
                    metadata = store_args[0][1]
                    
                    required_fields = ["file_id", "filename", "modality", "claim_id", "path", "file_size", "upload_timestamp"]
                    for field in required_fields:
                        assert field in metadata, f"Missing required field: {field}"
                    
                    # Verify field values
                    assert metadata["filename"] == "test_video.mp4"
                    assert metadata["modality"] == "video"
                    assert metadata["claim_id"] == "test-claim-metadata"
                    assert metadata["description"] == "Test description"
                    assert metadata["file_size"] > 0
                    assert "T" in metadata["upload_timestamp"]  # ISO format
                    assert "Z" in metadata["upload_timestamp"]  # UTC timezone
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
