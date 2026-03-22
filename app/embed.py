"""
Embedding service for multimodal content using Google Gemini Embedding 2 API.

This module provides a wrapper around the Gemini Embedding 2 API to generate
embeddings for different media types (video, image, audio, documents, text).
All embeddings are projected into a unified vector space for cross-modal search.
"""
import os
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)

# Embedding model configuration
# Using gemini-embedding-001 for text and gemini-embedding-2-preview for multimodal
EMBEDDING_MODEL = "models/gemini-embedding-001"  # For text
MULTIMODAL_EMBEDDING_MODEL = "models/gemini-embedding-2-preview"  # For multimodal

# Task types for different use cases
# RETRIEVAL_DOCUMENT: For embedding content that will be stored and retrieved
# RETRIEVAL_QUERY: For embedding search queries
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"  # For files being ingested
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"  # For search queries


async def embed_text(text: str, task_type: str = QUERY_TASK_TYPE) -> list[float]:
    """
    Generate embedding for text content.
    
    Args:
        text: The text content to embed
        task_type: The task type (RETRIEVAL_QUERY or RETRIEVAL_DOCUMENT)
    
    Returns:
        Embedding vector as list of floats
    
    Raises:
        Exception: If the Gemini API call fails
    
    Requirements: 4.1
    """
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        raise Exception(f"Failed to generate text embedding: {str(e)}")


async def embed_video(file_path: str) -> list[float]:
    """
    Generate embedding for video file.
    Uses Gemini Embedding 2 video modality.
    
    Args:
        file_path: Path to the video file
    
    Returns:
        Embedding vector as list of floats
    
    Raises:
        Exception: If the Gemini API call fails
    
    Requirements: 2.1
    """
    try:
        with open(file_path, 'rb') as f:
            video_bytes = f.read()
        
        # Determine MIME type from file extension
        mime_type = _get_video_mime_type(file_path)
        
        result = genai.embed_content(
            model=MULTIMODAL_EMBEDDING_MODEL,
            content={
                "parts": [{
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": video_bytes
                    }
                }]
            }
        )
        return result['embedding']
    except Exception as e:
        raise Exception(f"Failed to generate video embedding: {str(e)}")


async def embed_image(file_path: str) -> list[float]:
    """
    Generate embedding for image file.
    Uses Gemini Embedding 2 image modality.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        Embedding vector as list of floats
    
    Raises:
        Exception: If the Gemini API call fails
    
    Requirements: 2.2
    """
    try:
        with open(file_path, 'rb') as f:
            image_bytes = f.read()
        
        # Determine MIME type from file extension
        mime_type = _get_image_mime_type(file_path)
        
        result = genai.embed_content(
            model=MULTIMODAL_EMBEDDING_MODEL,
            content={
                "parts": [{
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": image_bytes
                    }
                }]
            }
        )
        return result['embedding']
    except Exception as e:
        raise Exception(f"Failed to generate image embedding: {str(e)}")


async def embed_audio(file_path: str) -> list[float]:
    """
    Generate embedding for audio file.
    Uses Gemini Embedding 2 audio modality.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Embedding vector as list of floats
    
    Raises:
        Exception: If the Gemini API call fails
    
    Requirements: 2.3
    """
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Determine MIME type from file extension
        mime_type = _get_audio_mime_type(file_path)
        
        result = genai.embed_content(
            model=MULTIMODAL_EMBEDDING_MODEL,
            content={
                "parts": [{
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": audio_bytes
                    }
                }]
            }
        )
        return result['embedding']
    except Exception as e:
        raise Exception(f"Failed to generate audio embedding: {str(e)}")


async def embed_document(file_path: str) -> list[float]:
    """
    Generate embedding for PDF document.
    Uses Gemini Embedding 2 document modality.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Embedding vector as list of floats
    
    Raises:
        Exception: If the Gemini API call fails
    
    Requirements: 2.4
    """
    try:
        with open(file_path, 'rb') as f:
            pdf_bytes = f.read()
        
        result = genai.embed_content(
            model=MULTIMODAL_EMBEDDING_MODEL,
            content={
                "parts": [{
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": pdf_bytes
                    }
                }]
            }
        )
        return result['embedding']
    except Exception as e:
        raise Exception(f"Failed to generate document embedding: {str(e)}")


def _get_video_mime_type(file_path: str) -> str:
    """
    Determine MIME type for video files based on extension.
    
    Args:
        file_path: Path to the video file
    
    Returns:
        MIME type string
    """
    extension = file_path.lower().split('.')[-1]
    mime_types = {
        'mp4': 'video/mp4',
        'mov': 'video/quicktime',
        'avi': 'video/x-msvideo',
        'webm': 'video/webm'
    }
    return mime_types.get(extension, 'video/mp4')


def _get_image_mime_type(file_path: str) -> str:
    """
    Determine MIME type for image files based on extension.
    
    Args:
        file_path: Path to the image file
    
    Returns:
        MIME type string
    """
    extension = file_path.lower().split('.')[-1]
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    return mime_types.get(extension, 'image/jpeg')


def _get_audio_mime_type(file_path: str) -> str:
    """
    Determine MIME type for audio files based on extension.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        MIME type string
    """
    extension = file_path.lower().split('.')[-1]
    mime_types = {
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'ogg': 'audio/ogg',
        'm4a': 'audio/mp4'
    }
    return mime_types.get(extension, 'audio/mpeg')


async def embed_file(file_path: str, modality: str) -> list[float]:
    """
    Unified interface that routes to appropriate embedding function based on modality.
    
    This function provides a single entry point for embedding any supported file type,
    automatically routing to the correct modality-specific function.
    
    Args:
        file_path: Path to the file to embed
        modality: The modality type ("video", "image", "audio", or "document")
    
    Returns:
        Embedding vector as list of floats
    
    Raises:
        ValueError: If the modality is not supported
        Exception: If the Gemini API call fails (propagated from modality-specific functions)
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    # Route to appropriate embedding function based on modality
    if modality == "video":
        return await embed_video(file_path)
    elif modality == "image":
        return await embed_image(file_path)
    elif modality == "audio":
        return await embed_audio(file_path)
    elif modality == "document":
        return await embed_document(file_path)
    else:
        raise ValueError(
            f"Unsupported modality: {modality}. "
            f"Supported modalities are: video, image, audio, document"
        )
