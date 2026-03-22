"""
File ingestion service for multimodal claims investigator.
Handles file validation, storage, embedding generation, and database operations.
"""
from fastapi import UploadFile
from typing import Tuple
import os
import shutil
from pathlib import Path
from datetime import datetime
import uuid

from app.embed import embed_file
from app.db import store_embedding

# Supported file extensions whitelist
SUPPORTED_EXTENSIONS = {
    "video": [".mp4", ".mov", ".avi", ".mkv"],
    "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
    "audio": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
    "document": [".pdf"]
}

# Maximum file size limit (100 MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB in bytes


def determine_modality(filename: str) -> str:
    """
    Determine modality (video/image/audio/document) from file extension.
    
    Args:
        filename: The name of the file including extension
        
    Returns:
        The modality string: "video", "image", "audio", or "document"
        
    Raises:
        ValueError: If the file extension is not supported
    """
    # Get file extension (lowercase)
    _, ext = os.path.splitext(filename.lower())
    
    # Check each modality for matching extension
    for modality, extensions in SUPPORTED_EXTENSIONS.items():
        if ext in extensions:
            return modality
    
    # If no match found, raise error
    raise ValueError(f"Unsupported file extension: {ext}")


def validate_file(file: UploadFile) -> Tuple[bool, str]:
    """
    Validate file type, size, and content.
    
    Performs the following checks:
    1. File extension is in the supported whitelist
    2. File size does not exceed maximum limit
    3. File is not empty
    
    Args:
        file: The uploaded file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all validations pass, False otherwise
        - error_message: Empty string if valid, descriptive error message if invalid
    """
    # Check if file has a filename
    if not file.filename:
        return False, "File must have a filename"
    
    # Validate file extension
    try:
        determine_modality(file.filename)
    except ValueError as e:
        # Get all supported extensions for error message
        all_extensions = []
        for extensions in SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(extensions)
        return False, f"Unsupported file type. Supported types: {', '.join(sorted(all_extensions))}"
    
    # Validate file size
    # Note: file.size might not be available in all cases, so we need to check
    if hasattr(file, 'size') and file.size is not None:
        if file.size == 0:
            return False, "File cannot be empty"
        if file.size > MAX_FILE_SIZE:
            max_size_mb = MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File size exceeds maximum limit of {max_size_mb:.0f} MB"
    
    # All validations passed
    return True, ""


async def save_file(file: UploadFile, claim_id: str) -> str:
    """
    Save file to disk with organized directory structure.
    
    Creates directory structure: uploads/{claim_id}/{modality}/
    Saves the file and returns the path.
    
    Args:
        file: The uploaded file to save
        claim_id: The claim ID to organize files by (can be empty string)
        
    Returns:
        The file path where the file was saved (relative to project root)
        
    Raises:
        ValueError: If file validation fails or modality cannot be determined
        IOError: If file cannot be saved to disk
    """
    # Determine the modality from filename
    modality = determine_modality(file.filename)
    
    # Use "unclaimed" if no claim_id provided
    claim_dir = claim_id if claim_id else "unclaimed"
    
    # Build directory path: uploads/{claim_id}/{modality}/
    dir_path = Path("uploads") / claim_dir / modality
    
    # Create directories if they don't exist
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Build full file path
    file_path = dir_path / file.filename
    
    # Save file to disk
    try:
        # Reset file pointer to beginning
        await file.seek(0)
        
        # Write file contents
        with open(file_path, "wb") as buffer:
            # Read and write in chunks to handle large files
            chunk_size = 1024 * 1024  # 1 MB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        
        # Return the path as a string (relative to project root)
        return str(file_path)
        
    except Exception as e:
        # Clean up partial file if write failed
        if file_path.exists():
            file_path.unlink()
        raise IOError(f"Failed to save file: {str(e)}")


async def ingest_file(file: UploadFile, claim_id: str = "", description: str = "") -> dict:
    """
    Main ingestion handler that orchestrates the complete file ingestion pipeline.
    
    This function performs the following steps:
    1. Validate the file (type, size, content)
    2. Save the file to disk with organized directory structure
    3. Generate embedding using Gemini Embedding 2 API
    4. Store embedding and metadata in ChromaDB
    
    The function ensures atomic behavior: if any step fails, previously completed
    steps are rolled back (e.g., uploaded file is deleted if embedding fails).
    
    Args:
        file: The uploaded file to ingest
        claim_id: Optional claim ID to associate with the file (default: "")
        description: Optional human-readable description (default: "")
    
    Returns:
        dict: Success response containing:
            - file_id: Unique identifier for the ingested file
            - filename: Original filename
            - modality: Detected modality (video/image/audio/document)
            - claim_id: Associated claim ID
            - status: "success"
    
    Raises:
        ValueError: If file validation fails
        Exception: If any step in the pipeline fails
    
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 3.1, 3.4, 9.1, 9.2, 9.3
    """
    file_path = None
    
    try:
        # Step 1: Validate the file
        is_valid, error_message = validate_file(file)
        if not is_valid:
            raise ValueError(error_message)
        
        # Step 2: Save file to disk
        file_path = await save_file(file, claim_id)
        
        # Determine modality for embedding
        modality = determine_modality(file.filename)
        
        # Step 3: Generate embedding
        embedding = await embed_file(file_path, modality)
        
        # Step 4: Prepare metadata
        file_id = str(uuid.uuid4())
        
        # Get file size from disk (more reliable than UploadFile.size)
        file_size = os.path.getsize(file_path)
        
        metadata = {
            "file_id": file_id,
            "filename": file.filename,
            "modality": modality,
            "claim_id": claim_id,
            "path": file_path,
            "file_size": file_size,
            "upload_timestamp": datetime.utcnow().isoformat() + "Z",
            "description": description
        }
        
        # Step 5: Store embedding and metadata in ChromaDB
        store_embedding(embedding, metadata, file_id)
        
        # Return success response
        return {
            "file_id": file_id,
            "filename": file.filename,
            "modality": modality,
            "claim_id": claim_id,
            "status": "success"
        }
    
    except Exception as e:
        # Atomic behavior: cleanup on failure
        # If file was saved but subsequent steps failed, delete the file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                # Log cleanup error but don't mask the original error
                print(f"Warning: Failed to cleanup file {file_path}: {cleanup_error}")
        
        # Re-raise the original exception
        raise
