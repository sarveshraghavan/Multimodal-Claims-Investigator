"""
Search service for semantic search across multimodal evidence files.

This module provides functions to:
- Convert ChromaDB distances to similarity scores
- Embed search queries and retrieve relevant results
- Format search results with metadata and scores
"""
from typing import Optional
from pydantic import BaseModel

from app.embed import embed_text, QUERY_TASK_TYPE
from app.db import search_embeddings


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str
    top_k: int = 10
    claim_id: str = ""


class SearchResult(BaseModel):
    """Individual search result with metadata and similarity score."""
    file_id: str
    filename: str
    modality: str
    claim_id: str
    similarity: float
    metadata: dict


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: list[SearchResult]
    query: str
    total_results: int


def calculate_similarity(distance: float) -> float:
    """
    Convert ChromaDB distance to similarity score (0-1).
    
    ChromaDB returns cosine distance (1 - cosine_similarity).
    We convert this to a similarity score where:
    - 0 distance = 1.0 similarity (identical)
    - Higher distance = lower similarity
    
    Uses formula: similarity = 1 / (1 + distance)
    This ensures:
    - Distance 0 → similarity 1.0
    - Distance 1 → similarity 0.5
    - Distance approaches infinity → similarity approaches 0
    
    Args:
        distance: The distance value from ChromaDB (non-negative float)
    
    Returns:
        Similarity score between 0 and 1 (higher = more similar)
    
    Requirements: 4.4
    """
    return 1.0 / (1.0 + distance)


async def search(query: str, top_k: int = 10, claim_id: str = "") -> dict:
    """
    Main search handler that orchestrates query embedding and database search.
    
    This function performs the following steps:
    1. Validate the query (non-empty)
    2. Embed the query text using Gemini Embedding 2 API
    3. Search ChromaDB for similar embeddings
    4. Convert distances to similarity scores
    5. Format and return results with metadata
    
    Args:
        query: The text query to search for
        top_k: Maximum number of results to return (default: 10)
        claim_id: Optional claim_id to filter results (default: "" for no filter)
    
    Returns:
        dict: Search response containing:
            - results: List of search results with metadata and similarity scores
            - query: The original query text
            - total_results: Number of results returned
    
    Raises:
        ValueError: If query is empty or top_k is invalid
        Exception: If embedding or search fails
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Validate query
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    # Validate top_k
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    
    # Step 1: Embed the query text
    query_embedding = await embed_text(query, task_type=QUERY_TASK_TYPE)
    
    # Step 2: Search ChromaDB for similar embeddings
    search_results = search_embeddings(
        query_embedding=query_embedding,
        top_k=top_k,
        claim_id=claim_id
    )
    
    # Step 3: Format results with similarity scores
    results = []
    ids = search_results.get("ids", [])
    metadatas = search_results.get("metadatas", [])
    distances = search_results.get("distances", [])
    
    for i in range(len(ids)):
        file_id = ids[i]
        metadata = metadatas[i]
        distance = distances[i]
        
        # Convert distance to similarity score
        similarity = calculate_similarity(distance)
        
        # Create search result
        result = {
            "file_id": file_id,
            "filename": metadata.get("filename", ""),
            "modality": metadata.get("modality", ""),
            "claim_id": metadata.get("claim_id", ""),
            "similarity": round(similarity, 4),  # Round to 4 decimal places
            "metadata": metadata
        }
        results.append(result)
    
    # Return formatted response
    return {
        "results": results,
        "query": query,
        "total_results": len(results)
    }
