"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import os
from dotenv import load_dotenv

from app.ingest import ingest_file
from app.search import search, SearchRequest
from app.retrieval import investigate
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()

# Verify GEMINI_API_KEY is set (allow test key for testing)
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    import warnings
    warnings.warn("GEMINI_API_KEY environment variable is not set. Some features will not work.")

# Custom exception classes
class NotFoundException(Exception):
    """Exception raised when a resource is not found."""
    pass


app = FastAPI(
    title="Multimodal Claims Investigator",
    description="Semantic search and investigation across videos, images, audio, and PDFs",
    version="1.0.0"
)


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors from Pydantic models and FastAPI request validation.
    Returns 400 Bad Request with error details.
    
    Requirements: 5.3, 10.1, 10.2, 10.3
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error",
            "error_type": "ValidationError",
            "details": exc.errors()
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Handle ValueError exceptions (typically from business logic validation).
    Returns 400 Bad Request with error message.
    
    Requirements: 5.3, 6.4, 10.1, 10.2, 10.3
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "error_type": "ValidationError"
        }
    )


@app.exception_handler(NotFoundException)
async def not_found_handler(request: Request, exc: NotFoundException):
    """
    Handle NotFoundException for missing resources.
    Returns 404 Not Found with error message.
    
    Requirements: 5.3
    """
    return JSONResponse(
        status_code=404,
        content={
            "error": str(exc),
            "error_type": "NotFoundError"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all other unhandled exceptions.
    Returns 500 Internal Server Error with error message.
    
    Requirements: 5.5, 10.4, 10.5, 10.6
    """
    # Log the full exception for debugging (in production, use proper logging)
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": f"Internal server error: {str(exc)}",
            "error_type": "InternalError"
        }
    )


# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "multimodal-claims-investigator"}

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "gemini_api_configured": bool(os.getenv("GEMINI_API_KEY"))
    }


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(...),
    claim_id: str = Form(""),
    description: str = Form("")
):
    """
    Ingest a media file for semantic search and investigation.
    
    Accepts multipart form data containing:
    - file: The media file to ingest (video, image, audio, or PDF)
    - claim_id: Optional claim ID to associate with the file
    - description: Optional human-readable description
    
    Returns:
        200: Success response with file metadata
        400: Validation error with details
        500: Internal server error
    
    Requirements: 5.1, 5.3, 5.4
    """
    result = await ingest_file(file, claim_id, description)
    return result


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """
    Search for media files using natural language text queries.
    
    Accepts JSON payload containing:
    - query: The text query to search for (required)
    - top_k: Maximum number of results to return (default: 10)
    - claim_id: Optional claim ID to filter results (default: "")
    
    Returns:
        200: Success response with search results
        400: Validation error with details
        500: Internal server error
    
    Requirements: 5.2, 5.3, 5.4
    """
    result = await search(
        query=request.query,
        top_k=request.top_k,
        claim_id=request.claim_id
    )
    return result


class InvestigationRequest(BaseModel):
    """Request model for investigation endpoint."""
    question: str
    claim_id: str = ""
    top_k: int = 6


@app.post("/investigate")
async def investigate_endpoint(request: InvestigationRequest):
    """
    Investigate a claim using AI-powered multimodal analysis.
    
    Accepts JSON payload containing:
    - question: The natural language question to investigate (required)
    - claim_id: Optional claim ID to filter evidence (default: "")
    - top_k: Maximum number of evidence files to analyze (default: 6)
    
    Returns:
        200: Success response with investigation report
        400: Validation error with details
        500: Internal server error
    
    Requirements: 5.2, 5.3, 5.4, 10.6
    """
    result = await investigate(
        question=request.question,
        claim_id=request.claim_id,
        top_k=request.top_k
    )
    return result
