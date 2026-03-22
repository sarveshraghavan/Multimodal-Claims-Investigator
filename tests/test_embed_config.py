"""
Tests for Gemini Embedding 2 API client configuration.
"""
import pytest
import os
from app.embed import (
    EMBEDDING_MODEL,
    EMBEDDING_TASK_TYPE,
    QUERY_TASK_TYPE,
    embed_text
)


def test_embedding_model_configured():
    """Verify EMBEDDING_MODEL is set to the correct model."""
    assert EMBEDDING_MODEL == "models/text-embedding-004"


def test_task_types_configured():
    """Verify task types are configured correctly."""
    assert EMBEDDING_TASK_TYPE == "RETRIEVAL_DOCUMENT"
    assert QUERY_TASK_TYPE == "RETRIEVAL_QUERY"


@pytest.mark.asyncio
async def test_embed_text_basic():
    """Test basic text embedding functionality."""
    # Skip if no API key configured
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not configured")
    
    text = "This is a test query"
    embedding = await embed_text(text)
    
    # Verify embedding is returned
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_embed_text_with_document_task_type():
    """Test text embedding with RETRIEVAL_DOCUMENT task type."""
    # Skip if no API key configured
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not configured")
    
    text = "This is a document to be stored"
    embedding = await embed_text(text, task_type=EMBEDDING_TASK_TYPE)
    
    # Verify embedding is returned
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) > 0
