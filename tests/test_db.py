"""
Unit tests for ChromaDB client and collection management.
"""
import pytest
import chromadb
from app.db import get_client, get_collection, COLLECTION_NAME


def test_get_client_returns_client():
    """Test that get_client() returns a ChromaDB client instance."""
    client = get_client()
    assert client is not None
    # Verify it has the expected client methods
    assert hasattr(client, 'get_or_create_collection')
    assert hasattr(client, 'list_collections')


def test_get_client_singleton():
    """Test that get_client() returns the same instance (singleton pattern)."""
    client1 = get_client()
    client2 = get_client()
    assert client1 is client2


def test_get_collection_returns_collection():
    """Test that get_collection() returns a ChromaDB collection."""
    collection = get_collection()
    assert collection is not None
    assert isinstance(collection, chromadb.Collection)
    assert collection.name == COLLECTION_NAME


def test_get_collection_uses_cosine_similarity():
    """Test that the collection is configured with cosine similarity metric."""
    collection = get_collection()
    metadata = collection.metadata
    assert metadata is not None
    assert "hnsw:space" in metadata
    assert metadata["hnsw:space"] == "cosine"


def test_get_collection_idempotent():
    """Test that get_collection() returns the same collection on multiple calls."""
    collection1 = get_collection()
    collection2 = get_collection()
    assert collection1.name == collection2.name


def test_store_embedding_success():
    """Test that store_embedding() successfully stores an embedding with metadata."""
    from app.db import store_embedding
    from datetime import datetime
    
    # Create test data
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    file_id = "test-file-123"
    metadata = {
        "filename": "test_video.mp4",
        "modality": "video",
        "claim_id": "claim-456",
        "path": "/uploads/claim-456/video/test_video.mp4",
        "file_size": 1024000,
        "upload_timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    # Store the embedding
    result_id = store_embedding(embedding, metadata, file_id)
    
    # Verify the returned ID matches
    assert result_id == file_id
    
    # Verify the embedding was stored by retrieving it
    collection = get_collection()
    result = collection.get(ids=[file_id])
    
    assert result is not None
    assert len(result["ids"]) == 1
    assert result["ids"][0] == file_id
    assert result["metadatas"][0]["filename"] == "test_video.mp4"
    assert result["metadatas"][0]["modality"] == "video"
    assert result["metadatas"][0]["claim_id"] == "claim-456"


def test_store_embedding_missing_required_fields():
    """Test that store_embedding() raises ValueError when required metadata fields are missing."""
    from app.db import store_embedding
    
    embedding = [0.1, 0.2, 0.3]
    file_id = "test-file-456"
    
    # Missing 'upload_timestamp' and 'file_size'
    incomplete_metadata = {
        "filename": "test.jpg",
        "modality": "image",
        "claim_id": "claim-789",
        "path": "/uploads/claim-789/image/test.jpg"
    }
    
    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        store_embedding(embedding, incomplete_metadata, file_id)
    
    # Verify error message mentions missing fields
    error_message = str(exc_info.value)
    assert "Missing required metadata fields" in error_message
    assert "file_size" in error_message
    assert "upload_timestamp" in error_message


def test_store_embedding_returns_unique_id():
    """Test that store_embedding() returns a unique identifier for each stored embedding."""
    from app.db import store_embedding
    from datetime import datetime
    
    # Store two different embeddings with the same dimensionality
    embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    file_id1 = "unique-file-1"
    metadata1 = {
        "filename": "file1.mp4",
        "modality": "video",
        "claim_id": "claim-001",
        "path": "/uploads/claim-001/video/file1.mp4",
        "file_size": 1000,
        "upload_timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    embedding2 = [0.6, 0.7, 0.8, 0.9, 1.0]
    file_id2 = "unique-file-2"
    metadata2 = {
        "filename": "file2.jpg",
        "modality": "image",
        "claim_id": "claim-002",
        "path": "/uploads/claim-002/image/file2.jpg",
        "file_size": 2000,
        "upload_timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    result_id1 = store_embedding(embedding1, metadata1, file_id1)
    result_id2 = store_embedding(embedding2, metadata2, file_id2)
    
    # Verify both IDs are returned and are different
    assert result_id1 == file_id1
    assert result_id2 == file_id2
    assert result_id1 != result_id2



def test_search_embeddings_returns_results():
    """Test that search_embeddings() returns similar embeddings."""
    from app.db import store_embedding, search_embeddings
    from datetime import datetime
    
    # Store some test embeddings
    embeddings_data = [
        ([0.1, 0.2, 0.3, 0.4, 0.5], "file-1", "video1.mp4", "claim-100"),
        ([0.2, 0.3, 0.4, 0.5, 0.6], "file-2", "image1.jpg", "claim-100"),
        ([0.9, 0.8, 0.7, 0.6, 0.5], "file-3", "audio1.mp3", "claim-200"),
    ]
    
    for embedding, file_id, filename, claim_id in embeddings_data:
        metadata = {
            "filename": filename,
            "modality": "video" if "video" in filename else "image" if "image" in filename else "audio",
            "claim_id": claim_id,
            "path": f"/uploads/{claim_id}/{filename}",
            "file_size": 1000,
            "upload_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        store_embedding(embedding, metadata, file_id)
    
    # Search with a query similar to the first embedding
    query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55]
    results = search_embeddings(query_embedding, top_k=3)
    
    # Verify results structure
    assert "ids" in results
    assert "metadatas" in results
    assert "distances" in results
    
    # Verify we got results
    assert len(results["ids"]) > 0
    assert len(results["metadatas"]) > 0
    assert len(results["distances"]) > 0
    
    # Verify all lists have the same length
    assert len(results["ids"]) == len(results["metadatas"])
    assert len(results["ids"]) == len(results["distances"])


def test_search_embeddings_respects_top_k():
    """Test that search_embeddings() respects the top_k parameter."""
    from app.db import store_embedding, search_embeddings
    from datetime import datetime
    
    # Store multiple embeddings
    for i in range(5):
        embedding = [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]
        file_id = f"topk-file-{i}"
        metadata = {
            "filename": f"file{i}.mp4",
            "modality": "video",
            "claim_id": "claim-topk",
            "path": f"/uploads/claim-topk/file{i}.mp4",
            "file_size": 1000,
            "upload_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        store_embedding(embedding, metadata, file_id)
    
    # Search with top_k=2
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = search_embeddings(query_embedding, top_k=2)
    
    # Verify we got at most 2 results
    assert len(results["ids"]) <= 2
    assert len(results["metadatas"]) <= 2
    assert len(results["distances"]) <= 2


def test_search_embeddings_filters_by_claim_id():
    """Test that search_embeddings() filters results by claim_id when provided."""
    from app.db import store_embedding, search_embeddings
    from datetime import datetime
    
    # Store embeddings with different claim_ids
    embeddings_data = [
        ([0.1, 0.2, 0.3, 0.4, 0.5], "filter-file-1", "video1.mp4", "claim-filter-A"),
        ([0.2, 0.3, 0.4, 0.5, 0.6], "filter-file-2", "image1.jpg", "claim-filter-A"),
        ([0.15, 0.25, 0.35, 0.45, 0.55], "filter-file-3", "audio1.mp3", "claim-filter-B"),
    ]
    
    for embedding, file_id, filename, claim_id in embeddings_data:
        metadata = {
            "filename": filename,
            "modality": "video" if "video" in filename else "image" if "image" in filename else "audio",
            "claim_id": claim_id,
            "path": f"/uploads/{claim_id}/{filename}",
            "file_size": 1000,
            "upload_timestamp": datetime.utcnow().isoformat() + "Z"
        }
        store_embedding(embedding, metadata, file_id)
    
    # Search with claim_id filter
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = search_embeddings(query_embedding, top_k=10, claim_id="claim-filter-A")
    
    # Verify all results belong to claim-filter-A
    assert len(results["metadatas"]) > 0
    for metadata in results["metadatas"]:
        assert metadata["claim_id"] == "claim-filter-A"


def test_search_embeddings_returns_metadata_and_distances():
    """Test that search_embeddings() returns complete metadata and distance scores."""
    from app.db import store_embedding, search_embeddings
    from datetime import datetime
    
    # Store a test embedding
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    file_id = "metadata-test-file"
    metadata = {
        "filename": "test_video.mp4",
        "modality": "video",
        "claim_id": "claim-metadata",
        "path": "/uploads/claim-metadata/video/test_video.mp4",
        "file_size": 5000,
        "upload_timestamp": datetime.utcnow().isoformat() + "Z"
    }
    store_embedding(embedding, metadata, file_id)
    
    # Search for it
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = search_embeddings(query_embedding, top_k=1)
    
    # Verify metadata is complete
    assert len(results["metadatas"]) > 0
    result_metadata = results["metadatas"][0]
    assert "filename" in result_metadata
    assert "modality" in result_metadata
    assert "claim_id" in result_metadata
    assert "path" in result_metadata
    assert "file_size" in result_metadata
    assert "upload_timestamp" in result_metadata
    
    # Verify distances are present
    assert len(results["distances"]) > 0
    assert isinstance(results["distances"][0], (int, float))


def test_search_embeddings_empty_results():
    """Test that search_embeddings() returns empty results when no matches found."""
    from app.db import search_embeddings
    
    # Search with a claim_id that doesn't exist
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = search_embeddings(query_embedding, top_k=10, claim_id="nonexistent-claim-xyz")
    
    # Verify empty results
    assert results["ids"] == []
    assert results["metadatas"] == []
    assert results["distances"] == []
