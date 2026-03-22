"""
Tests for modality-specific embedding functions.
"""
import pytest
from unittest.mock import patch, mock_open, MagicMock
from app.embed import (
    embed_video,
    embed_image,
    embed_audio,
    embed_document,
    embed_file,
    _get_video_mime_type,
    _get_image_mime_type,
    _get_audio_mime_type
)


# Test MIME type helper functions
def test_get_video_mime_type():
    """Test video MIME type detection."""
    assert _get_video_mime_type("video.mp4") == "video/mp4"
    assert _get_video_mime_type("video.mov") == "video/quicktime"
    assert _get_video_mime_type("video.MP4") == "video/mp4"  # Case insensitive
    assert _get_video_mime_type("video.unknown") == "video/mp4"  # Default


def test_get_image_mime_type():
    """Test image MIME type detection."""
    assert _get_image_mime_type("image.jpg") == "image/jpeg"
    assert _get_image_mime_type("image.jpeg") == "image/jpeg"
    assert _get_image_mime_type("image.png") == "image/png"
    assert _get_image_mime_type("image.PNG") == "image/png"  # Case insensitive
    assert _get_image_mime_type("image.unknown") == "image/jpeg"  # Default


def test_get_audio_mime_type():
    """Test audio MIME type detection."""
    assert _get_audio_mime_type("audio.mp3") == "audio/mpeg"
    assert _get_audio_mime_type("audio.wav") == "audio/wav"
    assert _get_audio_mime_type("audio.MP3") == "audio/mpeg"  # Case insensitive
    assert _get_audio_mime_type("audio.unknown") == "audio/mpeg"  # Default


# Test embedding functions with mocked API calls
@pytest.mark.asyncio
@patch('app.embed.genai.embed_content')
@patch('builtins.open', new_callable=mock_open, read_data=b'fake_video_data')
async def test_embed_video(mock_file, mock_embed):
    """Test video embedding function."""
    # Mock the API response
    mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}
    
    result = await embed_video("test_video.mp4")
    
    # Verify file was opened
    mock_file.assert_called_once_with("test_video.mp4", 'rb')
    
    # Verify API was called with correct parameters
    mock_embed.assert_called_once()
    call_args = mock_embed.call_args
    assert call_args[1]['model'] == 'models/gemini-embedding-2-preview'
    assert 'content' in call_args[1]
    
    # Verify result
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@patch('app.embed.genai.embed_content')
@patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
async def test_embed_image(mock_file, mock_embed):
    """Test image embedding function."""
    # Mock the API response
    mock_embed.return_value = {'embedding': [0.4, 0.5, 0.6]}
    
    result = await embed_image("test_image.jpg")
    
    # Verify file was opened
    mock_file.assert_called_once_with("test_image.jpg", 'rb')
    
    # Verify API was called
    mock_embed.assert_called_once()
    call_args = mock_embed.call_args
    assert call_args[1]['model'] == 'models/gemini-embedding-2-preview'
    
    # Verify result
    assert result == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
@patch('app.embed.genai.embed_content')
@patch('builtins.open', new_callable=mock_open, read_data=b'fake_audio_data')
async def test_embed_audio(mock_file, mock_embed):
    """Test audio embedding function."""
    # Mock the API response
    mock_embed.return_value = {'embedding': [0.7, 0.8, 0.9]}
    
    result = await embed_audio("test_audio.mp3")
    
    # Verify file was opened
    mock_file.assert_called_once_with("test_audio.mp3", 'rb')
    
    # Verify API was called
    mock_embed.assert_called_once()
    call_args = mock_embed.call_args
    assert call_args[1]['model'] == 'models/gemini-embedding-2-preview'
    
    # Verify result
    assert result == [0.7, 0.8, 0.9]


@pytest.mark.asyncio
@patch('app.embed.genai.embed_content')
@patch('builtins.open', new_callable=mock_open, read_data=b'fake_pdf_data')
async def test_embed_document(mock_file, mock_embed):
    """Test document (PDF) embedding function."""
    # Mock the API response
    mock_embed.return_value = {'embedding': [1.0, 1.1, 1.2]}
    
    result = await embed_document("test_document.pdf")
    
    # Verify file was opened
    mock_file.assert_called_once_with("test_document.pdf", 'rb')
    
    # Verify API was called with PDF MIME type
    mock_embed.assert_called_once()
    call_args = mock_embed.call_args
    assert call_args[1]['model'] == 'models/gemini-embedding-2-preview'
    content = call_args[1]['content']
    assert content['parts'][0]['inline_data']['mime_type'] == 'application/pdf'
    
    # Verify result
    assert result == [1.0, 1.1, 1.2]


@pytest.mark.asyncio
@patch('app.embed.genai.embed_content')
@patch('builtins.open', side_effect=FileNotFoundError("File not found"))
async def test_embed_video_file_not_found(mock_file, mock_embed):
    """Test video embedding with missing file."""
    with pytest.raises(Exception) as exc_info:
        await embed_video("nonexistent.mp4")
    
    assert "Failed to generate video embedding" in str(exc_info.value)


@pytest.mark.asyncio
@patch('app.embed.genai.embed_content', side_effect=Exception("API Error"))
@patch('builtins.open', new_callable=mock_open, read_data=b'fake_data')
async def test_embed_image_api_error(mock_file, mock_embed):
    """Test image embedding with API error."""
    with pytest.raises(Exception) as exc_info:
        await embed_image("test.jpg")
    
    assert "Failed to generate image embedding" in str(exc_info.value)


# Test unified embed_file interface
@pytest.mark.asyncio
@patch('app.embed.embed_video')
async def test_embed_file_routes_to_video(mock_embed_video):
    """Test that embed_file routes to embed_video for video modality."""
    mock_embed_video.return_value = [0.1, 0.2, 0.3]
    
    result = await embed_file("test.mp4", "video")
    
    mock_embed_video.assert_called_once_with("test.mp4")
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@patch('app.embed.embed_image')
async def test_embed_file_routes_to_image(mock_embed_image):
    """Test that embed_file routes to embed_image for image modality."""
    mock_embed_image.return_value = [0.4, 0.5, 0.6]
    
    result = await embed_file("test.jpg", "image")
    
    mock_embed_image.assert_called_once_with("test.jpg")
    assert result == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
@patch('app.embed.embed_audio')
async def test_embed_file_routes_to_audio(mock_embed_audio):
    """Test that embed_file routes to embed_audio for audio modality."""
    mock_embed_audio.return_value = [0.7, 0.8, 0.9]
    
    result = await embed_file("test.mp3", "audio")
    
    mock_embed_audio.assert_called_once_with("test.mp3")
    assert result == [0.7, 0.8, 0.9]


@pytest.mark.asyncio
@patch('app.embed.embed_document')
async def test_embed_file_routes_to_document(mock_embed_document):
    """Test that embed_file routes to embed_document for document modality."""
    mock_embed_document.return_value = [1.0, 1.1, 1.2]
    
    result = await embed_file("test.pdf", "document")
    
    mock_embed_document.assert_called_once_with("test.pdf")
    assert result == [1.0, 1.1, 1.2]


@pytest.mark.asyncio
async def test_embed_file_unsupported_modality():
    """Test that embed_file raises ValueError for unsupported modality."""
    with pytest.raises(ValueError) as exc_info:
        await embed_file("test.txt", "text")
    
    assert "Unsupported modality: text" in str(exc_info.value)
    assert "video, image, audio, document" in str(exc_info.value)


@pytest.mark.asyncio
@patch('app.embed.embed_video', side_effect=Exception("API Error"))
async def test_embed_file_propagates_errors(mock_embed_video):
    """Test that embed_file propagates errors from modality-specific functions."""
    with pytest.raises(Exception) as exc_info:
        await embed_file("test.mp4", "video")
    
    assert "API Error" in str(exc_info.value)
