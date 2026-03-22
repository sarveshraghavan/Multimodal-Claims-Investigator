"""Unit tests for file ingestion service."""
import pytest
from fastapi import UploadFile
from io import BytesIO
from app.ingest import determine_modality, validate_file, MAX_FILE_SIZE, SUPPORTED_EXTENSIONS


class TestDetermineModality:
    def test_video_extensions(self):
        assert determine_modality("video.mp4") == "video"
        assert determine_modality("video.mov") == "video"
        assert determine_modality("video.avi") == "video"
    
    def test_image_extensions(self):
        assert determine_modality("image.jpg") == "image"
        assert determine_modality("image.png") == "image"
        assert determine_modality("image.gif") == "image"
    
    def test_audio_extensions(self):
        assert determine_modality("audio.mp3") == "audio"
        assert determine_modality("audio.wav") == "audio"
    
    def test_document_extensions(self):
        assert determine_modality("document.pdf") == "document"
    
    def test_case_insensitive(self):
        assert determine_modality("VIDEO.MP4") == "video"
        assert determine_modality("Image.JPG") == "image"
    
    def test_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            determine_modality("file.txt")


class TestValidateFile:
    def create_upload_file(self, filename, content=b"test", size=None):
        file = UploadFile(filename=filename, file=BytesIO(content))
        file.size = size if size is not None else len(content)
        return file
    
    def test_valid_video_file(self):
        file = self.create_upload_file("video.mp4", b"content", 1024)
        is_valid, error = validate_file(file)
        assert is_valid is True
        assert error == ""
    
    def test_valid_image_file(self):
        file = self.create_upload_file("image.jpg", b"content", 2048)
        is_valid, error = validate_file(file)
        assert is_valid is True
        assert error == ""
    
    def test_missing_filename(self):
        file = UploadFile(filename="", file=BytesIO(b"content"))
        is_valid, error = validate_file(file)
        assert is_valid is False
        assert "must have a filename" in error
    
    def test_unsupported_file_type(self):
        file = self.create_upload_file("document.txt", b"content", 1024)
        is_valid, error = validate_file(file)
        assert is_valid is False
        assert "Unsupported file type" in error
        assert "Supported types:" in error
    
    def test_empty_file(self):
        file = self.create_upload_file("video.mp4", b"", 0)
        is_valid, error = validate_file(file)
        assert is_valid is False
        assert "File cannot be empty" in error
    
    def test_file_exceeds_size_limit(self):
        file = self.create_upload_file("video.mp4", b"content", MAX_FILE_SIZE + 1)
        is_valid, error = validate_file(file)
        assert is_valid is False
        assert "exceeds maximum limit" in error
        assert "MB" in error
    
    def test_file_at_size_limit(self):
        file = self.create_upload_file("video.mp4", b"content", MAX_FILE_SIZE)
        is_valid, error = validate_file(file)
        assert is_valid is True
        assert error == ""
