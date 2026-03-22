"""
ChromaDB client and collection management for evidence storage.

This module provides functions to:
- Initialize and manage a persistent ChromaDB client
- Create and retrieve the evidence collection
- Store embeddings with metadata
- Search embeddings with optional filtering
"""
import chromadb
from chromadb.config import Settings
import os
from typing import Optional

# ChromaDB configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
COLLECTION_NAME = "evidence_files"

# Global client instance (singleton pattern)
_client: Optional[chromadb.Client] = None


def get_client() -> chromadb.Client:
    """
    Get or create a persistent ChromaDB client.
    
    Uses a singleton pattern to ensure only one client instance exists.
    The client is configured with persistent storage in CHROMA_PERSIST_DIR.
    
    Returns:
        chromadb.Client: The persistent ChromaDB client instance
    """
    global _client
    
    if _client is None:
        # Create persistent client with settings
        _client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
    
    return _client


def get_collection() -> chromadb.Collection:
    """
    Get or create the evidence collection for storing file embeddings.
    
    The collection is configured with:
    - Name: "evidence_files"
    - Distance metric: cosine similarity (default for ChromaDB)
    
    Returns:
        chromadb.Collection: The evidence files collection
    """
    client = get_client()
    
    # Get or create collection with cosine similarity metric
    # ChromaDB uses cosine distance by default (which is 1 - cosine_similarity)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Explicitly set cosine similarity
    )
    
    return collection


def store_embedding(
    embedding: list[float],
    metadata: dict,
    file_id: str
) -> str:
    """
    Store an embedding vector with metadata in ChromaDB.
    
    This function stores a single embedding vector along with its associated
    metadata in the evidence collection. All required metadata fields must
    be present: filename, modality, claim_id, path, file_size, upload_timestamp.
    
    Args:
        embedding: The embedding vector (list of floats)
        metadata: Dictionary containing required fields:
            - filename: Name of the file
            - modality: Type of media (video|image|audio|document)
            - claim_id: Associated claim identifier
            - path: File system path to the stored file
            - file_size: Size of the file in bytes
            - upload_timestamp: ISO 8601 timestamp of upload
        file_id: Unique identifier for this document
    
    Returns:
        str: The unique document ID assigned by ChromaDB (same as file_id)
    
    Raises:
        ValueError: If required metadata fields are missing
        Exception: If storage operation fails
    
    Requirements: 3.1, 3.2, 3.3
    """
    # Validate required metadata fields
    required_fields = ["filename", "modality", "claim_id", "path", "file_size", "upload_timestamp"]
    missing_fields = [field for field in required_fields if field not in metadata]
    
    if missing_fields:
        raise ValueError(f"Missing required metadata fields: {', '.join(missing_fields)}")
    
    # Get the collection
    collection = get_collection()
    
    # Store the embedding with metadata
    # ChromaDB expects:
    # - embeddings: list of vectors (we have one)
    # - metadatas: list of metadata dicts (we have one)
    # - ids: list of document IDs (we have one)
    collection.add(
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[file_id]
    )
    
    return file_id


def search_embeddings(
    query_embedding: list[float],
    top_k: int = 10,
    claim_id: str = ""
) -> dict:
    """
    Search for similar embeddings in the vector database.
    
    This function performs a vector similarity search using the provided query
    embedding. Results can be optionally filtered by claim_id. Returns the
    top-k most similar vectors with their metadata and distance scores.
    
    Args:
        query_embedding: The query vector to search for (list of floats)
        top_k: Maximum number of results to return (default: 10)
        claim_id: Optional claim_id to filter results (default: "" for no filter)
    
    Returns:
        dict: Search results containing:
            - ids: List of document IDs
            - metadatas: List of metadata dictionaries
            - distances: List of distance scores (lower = more similar)
            - embeddings: List of embedding vectors (if requested)
    
    Requirements: 4.2, 4.3, 4.4, 4.5
    """
    collection = get_collection()
    
    # Build where clause for claim_id filtering if provided
    where_clause = None
    if claim_id:
        where_clause = {"claim_id": claim_id}
    
    # Perform the query
    # ChromaDB query returns:
    # - ids: list of lists (one list per query)
    # - metadatas: list of lists
    # - distances: list of lists
    # - embeddings: list of lists (if include is specified)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
        include=["metadatas", "distances"]
    )
    
    # Flatten the results (ChromaDB returns lists of lists for batch queries)
    # Since we only have one query, we take the first element of each list
    return {
        "ids": results["ids"][0] if results["ids"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
        "distances": results["distances"][0] if results["distances"] else []
    }
