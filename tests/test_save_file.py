"""Unit tests for save_file function."""
import pytest
import os
import shutil
from fastapi import UploadFile
from io import BytesIO
from app.ingest import save_file


def create_upload_file(filename: str, content: bytes = b"test", size: int = None):
    """Helper to create mock UploadFile."""
    file = UploadFile(filename=filename, file=BytesIO(content))
    file.size = size if size is not None else len(content)
    return file


class TestSaveFile:
    """Tests for save_file function."""
    
    @pytest.mark.asyncio
    async def test_creates_directory_structure(self):
        """Test save_file creates organized directory structure."""
        # Clean up test directory if it exists
        test_dir = "uploads/test-claim-123"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            # Create test file
            file = create_upload_file("test_video.mp4", b"test video content")
            
            # Save file
            file_path = await save_file(file, "test-claim-123")
            
            # Verify path structure (handle both Windows and Unix paths)
            assert "uploads" in file_path
            assert "test-claim-123" in file_path
            assert "video" in file_path
            assert "test_video.mp4" in file_path
            
            # Verify directory exists
            assert os.path.exists("uploads/test-claim-123/video")
            
            # Verify file exists and has correct content
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                assert f.read() == b"test video content"
        
        finally:
            # Clean up
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_different_modalities(self):
        """Test save_file organizes files by modality."""
        test_dir = "uploads/test-claim-456"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            # Save files of different modalities
            video_file = create_upload_file("video.mp4", b"video")
            image_file = create_upload_file("image.jpg", b"image")
            audio_file = create_upload_file("audio.mp3", b"audio")
            doc_file = create_upload_file("doc.pdf", b"document")
            
            video_path = await save_file(video_file, "test-claim-456")
            image_path = await save_file(image_file, "test-claim-456")
            audio_path = await save_file(audio_file, "test-claim-456")
            doc_path = await save_file(doc_file, "test-claim-456")
            
            # Verify files exist
            assert os.path.exists(video_path)
            assert os.path.exists(image_path)
            assert os.path.exists(audio_path)
            assert os.path.exists(doc_path)
            
            # Verify correct modality directories
            assert "video" in video_path
            assert "image" in image_path
            assert "audio" in audio_path
            assert "document" in doc_path
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_no_claim_id(self):
        """Test save_file uses 'unclaimed' directory when no claim_id provided."""
        test_dir = "uploads/unclaimed"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            file = create_upload_file("test.jpg", b"test image")
            
            # Save with empty claim_id
            file_path = await save_file(file, "")
            
            # Verify uses 'unclaimed' directory
            assert "unclaimed" in file_path
            assert "image" in file_path
            assert os.path.exists(file_path)
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
    
    @pytest.mark.asyncio
    async def test_returns_path(self):
        """Test save_file returns the correct file path."""
        test_dir = "uploads/test-claim-789"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        try:
            file = create_upload_file("document.pdf", b"pdf content")
            
            file_path = await save_file(file, "test-claim-789")
            
            # Verify file can be read using returned path
            assert os.path.exists(file_path)
            with open(file_path, "rb") as f:
                assert f.read() == b"pdf content"
        
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
