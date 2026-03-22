"""
Investigation service for AI-powered multimodal analysis using Gemini 2.5 Pro.

This module provides functions to:
- Map file extensions to MIME types for Gemini API
- Load evidence files from disk and construct multimodal content
- Call Gemini 2.5 Pro for reasoning across multiple evidence types
- Orchestrate the complete investigation workflow
"""
import os
from typing import Optional
import google.generativeai as genai
from pathlib import Path
from dotenv import load_dotenv

from app.embed import embed_text, QUERY_TASK_TYPE
from app.db import search_embeddings

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY", "")
if api_key:
    genai.configure(api_key=api_key)

# Gemini 2.5 Pro model for multimodal reasoning
GEMINI_PRO_MODEL = "gemini-2.5-flash"

# System prompt for investigation
INVESTIGATION_SYSTEM_PROMPT = """You are an expert insurance claims investigator AI.
You will be given a question from a claims adjuster and a set of evidence files
(which may include dashcam video, damage photos, audio recordings, and PDF reports).

Your job:
1. Carefully analyse ALL provided evidence together.
2. Directly answer the adjuster's question with specific references to the evidence.
3. Note any inconsistencies between files (e.g. audio vs visual account mismatch).
4. Be concise, factual, and highlight the most important finding first.
5. If evidence is insufficient to answer, say so clearly.

Never make up facts. Only state what the evidence actually shows."""


def _mime_for_modality(modality: str, suffix: str) -> str:
    """
    Map file extension to MIME type for Gemini API.
    
    This function determines the appropriate MIME type based on the file's
    modality and extension, which is required for constructing multimodal
    content for the Gemini API.
    
    Args:
        modality: The modality type ("video", "image", "audio", or "document")
        suffix: The file extension (e.g., ".mp4", ".jpg", ".pdf")
    
    Returns:
        str: The MIME type string (e.g., "video/mp4", "image/jpeg")
    
    Requirements: 8.2
    """
    suffix = suffix.lower()
    
    if modality == "video":
        mime_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm"
        }
        return mime_types.get(suffix, "video/mp4")
    
    elif modality == "image":
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp"
        }
        return mime_types.get(suffix, "image/jpeg")
    
    elif modality == "audio":
        mime_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg"
        }
        return mime_types.get(suffix, "audio/mpeg")
    
    elif modality == "document":
        return "application/pdf"
    
    else:
        # Default fallback
        return "application/octet-stream"


def _call_gemini(question: str, metadatas: list, distances: list) -> tuple[str, list]:
    """
    Synchronous Gemini 2.5 Pro call for multimodal reasoning.
    
    This function constructs multimodal content from evidence files and sends
    them to Gemini 2.5 Pro for analysis. It handles:
    - Loading files from disk
    - Constructing inline data with appropriate MIME types
    - Building the prompt with question and evidence
    - Handling missing files gracefully
    
    Args:
        question: The natural language question from the claims adjuster
        metadatas: List of metadata dictionaries for evidence files
        distances: List of distance scores corresponding to each evidence file
    
    Returns:
        tuple: (answer, sources)
            - answer: The investigation report from Gemini 2.5 Pro
            - sources: List of source references that were successfully loaded
    
    Raises:
        Exception: If Gemini API call fails
    
    Requirements: 8.2, 8.8, 9.4, 9.5
    """
    # Build multimodal content parts
    content_parts = []
    sources = []
    
    # Add system prompt and question first
    content_parts.append(INVESTIGATION_SYSTEM_PROMPT)
    content_parts.append(f"\n\nQuestion: {question}\n\nEvidence Files:\n")
    
    # Load and add each evidence file
    for i, metadata in enumerate(metadatas):
        filename = metadata.get("filename", "unknown")
        file_path = metadata.get("path", "")
        modality = metadata.get("modality", "")
        claim_id = metadata.get("claim_id", "")
        
        # Calculate similarity score from distance
        from app.search import calculate_similarity
        similarity = calculate_similarity(distances[i])
        
        # Try to load the file
        if not file_path or not os.path.exists(file_path):
            # File not found - note it but continue
            content_parts.append(f"\n[Evidence {i+1}: {filename} - FILE NOT FOUND]")
            continue
        
        try:
            # Read file bytes
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Determine MIME type
            suffix = Path(file_path).suffix
            mime_type = _mime_for_modality(modality, suffix)
            
            # Add file label
            content_parts.append(f"\n[Evidence {i+1}: {filename} ({modality})]")
            
            # Add inline data
            content_parts.append({
                "mime_type": mime_type,
                "data": file_bytes
            })
            
            # Track this source
            sources.append({
                "filename": filename,
                "modality": modality,
                "claim_id": claim_id,
                "similarity": round(similarity, 4)
            })
            
        except Exception as e:
            # Error loading file - note it but continue
            content_parts.append(f"\n[Evidence {i+1}: {filename} - ERROR LOADING: {str(e)}]")
            continue
    
    # If no evidence files were successfully loaded, return early
    if not sources:
        return "Unable to analyze: No evidence files could be loaded.", []
    
    # Call Gemini 2.5 Pro
    try:
        model = genai.GenerativeModel(GEMINI_PRO_MODEL)
        response = model.generate_content(content_parts)
        answer = response.text
        
        return answer, sources
        
    except Exception as e:
        raise Exception(f"Failed to generate investigation report: {str(e)}")


async def investigate(question: str, claim_id: str = "", top_k: int = 6) -> dict:
    """
    Main investigation handler that orchestrates the complete investigation workflow.
    
    This function performs the following steps:
    1. Validate the question
    2. Embed the question using Gemini Embedding 2 API
    3. Retrieve top-k most relevant evidence files from ChromaDB
    4. Load evidence files from disk
    5. Send question and evidence to Gemini 2.5 Pro for multimodal analysis
    6. Format and return the investigation report with sources
    
    Args:
        question: The natural language question from the claims adjuster
        claim_id: Optional claim_id to filter evidence (default: "" for no filter)
        top_k: Maximum number of evidence files to retrieve (default: 6)
    
    Returns:
        dict: Investigation report containing:
            - answer: The AI-generated investigation report
            - sources: List of evidence files analyzed with similarity scores
            - model_used: The Gemini model used for analysis
    
    Raises:
        ValueError: If question is empty or top_k is invalid
        Exception: If any step in the pipeline fails
    
    Requirements: 8.1, 8.2, 8.5, 8.6, 8.8
    """
    # Validate question
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Validate top_k
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    
    # Step 1: Embed the question
    question_embedding = await embed_text(question, task_type=QUERY_TASK_TYPE)
    
    # Step 2: Retrieve relevant evidence files
    search_results = search_embeddings(
        query_embedding=question_embedding,
        top_k=top_k,
        claim_id=claim_id
    )
    
    metadatas = search_results.get("metadatas", [])
    distances = search_results.get("distances", [])
    
    # Check if any evidence was found
    if not metadatas:
        return {
            "answer": "No evidence files found to analyze. Please ingest relevant files first.",
            "sources": [],
            "model_used": GEMINI_PRO_MODEL
        }
    
    # Step 3: Call Gemini 2.5 Pro for analysis
    answer, sources = _call_gemini(question, metadatas, distances)
    
    # Step 4: Format and return response
    return {
        "answer": answer,
        "sources": sources,
        "model_used": GEMINI_PRO_MODEL
    }
