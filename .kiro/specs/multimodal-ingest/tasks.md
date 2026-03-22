# Implementation Plan: Multimodal Claims Investigator

## Overview

This implementation plan breaks down the multimodal claims investigator system into discrete, incremental coding tasks. The system will be built in layers: database and embedding services first, then ingestion, search, investigation, and finally the frontend. Each task builds on previous work, with property-based tests integrated throughout to validate correctness early.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create Python project with FastAPI, ChromaDB, google-generativeai, hypothesis, pytest
  - Create directory structure: `app/`, `tests/`, `uploads/`
  - Set up environment variables for GEMINI_API_KEY
  - Create requirements.txt with all dependencies
  - _Requirements: All (foundational)_

- [ ] 2. Implement database service (db.py)
  - [x] 2.1 Create ChromaDB client and collection management
    - Implement `get_client()` for persistent ChromaDB client
    - Implement `get_collection()` for evidence collection
    - Configure collection with cosine similarity metric
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 2.2 Implement vector storage function
    - Implement `store_embedding()` with metadata schema
    - Ensure all required metadata fields are stored (filename, modality, claim_id, path, file_size, upload_timestamp)
    - Return unique document ID
    - _Requirements: 3.1, 3.2, 3.3_
  
  - [x] 2.3 Implement vector search function
    - Implement `search_embeddings()` with top_k parameter
    - Support optional claim_id filtering
    - Return results with metadata and distances
    - _Requirements: 4.2, 4.3, 4.4, 4.5_
  
  - [ ]* 2.4 Write property test for embedding storage round-trip
    - **Property 4: Embedding Storage Round-Trip**
    - **Validates: Requirements 3.1**
  
  - [ ]* 2.5 Write property test for required metadata fields
    - **Property 5: Required Metadata Fields**
    - **Validates: Requirements 3.2**
  
  - [ ]* 2.6 Write property test for storage success returns identifier
    - **Property 6: Storage Success Returns Identifier**
    - **Validates: Requirements 3.3**

- [ ] 3. Implement embedding service (embed.py)
  - [x] 3.1 Configure Gemini Embedding 2 API client
    - Import and configure google.generativeai
    - Set up EMBEDDING_MODEL = "models/embedding-002"
    - Configure task types for documents vs queries
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  
  - [x] 3.2 Implement modality-specific embedding functions
    - Implement `embed_video()` for video files
    - Implement `embed_image()` for image files
    - Implement `embed_audio()` for audio files
    - Implement `embed_document()` for PDF files
    - Implement `embed_text()` for text queries
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 4.1_
  
  - [x] 3.3 Implement unified embedding interface
    - Implement `embed_file()` that routes to appropriate function based on modality
    - Handle API errors and propagate descriptive messages
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  
  - [ ]* 3.4 Write property test for consistent embedding dimensionality
    - **Property 3: Consistent Embedding Dimensionality**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.6**
  
  - [ ]* 3.5 Write property test for query embedding generation
    - **Property 7: Query Embedding Generation**
    - **Validates: Requirements 4.1**
  
  - [ ]* 3.6 Write unit test for API error propagation
    - Mock Gemini API failures
    - Verify descriptive error messages returned
    - _Requirements: 2.5_

- [ ] 4. Implement file ingestion service (ingest.py)
  - [x] 4.1 Implement file validation functions
    - Implement `validate_file()` for type, size, and content checks
    - Implement `determine_modality()` from file extension
    - Define supported file extensions whitelist
    - Define maximum file size limit
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 6.1, 6.2, 6.3, 6.4_
  
  - [x] 4.2 Implement file storage function
    - Implement `save_file()` with organized directory structure
    - Create directories: uploads/{claim_id}/{modality}/
    - Save file and return path
    - _Requirements: 9.1, 9.2_
  
  - [x] 4.3 Implement main ingestion handler
    - Implement `ingest_file()` orchestrating validation, storage, embedding, and database operations
    - Ensure atomic behavior: cleanup on failure
    - Return success response with file_id
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 3.1, 3.4, 9.1, 9.2, 9.3_
  
  - [x] 4.4 Create FastAPI endpoint for ingestion
    - Create POST /ingest endpoint accepting multipart form data
    - Accept file, claim_id (optional), description (optional)
    - Return 200 with file metadata on success
    - Return 400 with error details on validation failure
    - _Requirements: 5.1, 5.3, 5.4_
  
  - [ ]* 4.5 Write property test for multimodal file acceptance
    - **Property 1: Multimodal File Acceptance**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.6**
  
  - [ ]* 4.6 Write property test for unsupported file rejection
    - **Property 2: Unsupported File Rejection**
    - **Validates: Requirements 1.5**
  
  - [ ]* 4.7 Write property test for file size validation
    - **Property 12: File Size Validation**
    - **Validates: Requirements 6.1**
  
  - [ ]* 4.8 Write property test for file extension validation
    - **Property 13: File Extension Validation**
    - **Validates: Requirements 6.2**
  
  - [ ]* 4.9 Write property test for specific validation error messages
    - **Property 14: Specific Validation Error Messages**
    - **Validates: Requirements 6.4**
  
  - [ ]* 4.10 Write property test for file storage and retrieval round-trip
    - **Property 19: File Storage and Retrieval Round-Trip**
    - **Validates: Requirements 9.1, 9.2, 9.4**
  
  - [ ]* 4.11 Write property test for claim ID metadata inclusion
    - **Property 20: Claim ID Metadata Inclusion**
    - **Validates: Requirements 9.3**
  
  - [ ]* 4.12 Write unit test for atomic ingestion behavior
    - Test that failures trigger cleanup
    - Verify no partial data left on error
    - _Requirements: 3.4_

- [x] 5. Checkpoint - Verify ingestion pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement search service (search.py)
  - [x] 6.1 Implement similarity score calculation
    - Implement `calculate_similarity()` converting distance to score
    - Use formula: similarity = 1 / (1 + distance)
    - _Requirements: 4.4_
  
  - [x] 6.2 Implement main search handler
    - Implement `search()` orchestrating query embedding and database search
    - Format results with metadata and similarity scores
    - Support optional claim_id filtering
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 6.3 Create FastAPI endpoint for search
    - Create POST /search endpoint accepting JSON payload
    - Accept query, top_k (default 10), claim_id (optional)
    - Return 200 with results array
    - Return 400 on validation errors
    - _Requirements: 5.2, 5.3, 5.4_
  
  - [ ]* 6.4 Write property test for search result limit enforcement
    - **Property 8: Search Result Limit Enforcement**
    - **Validates: Requirements 4.2, 4.5**
  
  - [ ]* 6.5 Write property test for search results include complete information
    - **Property 9: Search Results Include Complete Information**
    - **Validates: Requirements 4.3, 4.4**
  
  - [ ]* 6.6 Write unit test for empty search results edge case
    - Test query with no matching results
    - Verify empty result set returned
    - _Requirements: 4.6_

- [ ] 7. Implement investigation service (retrieval.py)
  - [x] 7.1 Implement MIME type mapping
    - Implement `_mime_for_modality()` mapping extensions to MIME types
    - Support video, image, audio, and PDF formats
    - _Requirements: 8.2_
  
  - [x] 7.2 Implement Gemini 2.5 Pro call function
    - Implement `_call_gemini()` building multimodal content
    - Load evidence files from disk
    - Construct content with file labels and inline data
    - Handle missing files gracefully
    - Return answer and sources used
    - _Requirements: 8.2, 8.8, 9.4, 9.5_
  
  - [x] 7.3 Implement main investigation handler
    - Implement `investigate()` orchestrating question embedding, evidence retrieval, and LLM analysis
    - Support optional claim_id filtering
    - Support configurable top_k parameter
    - Format response with answer, sources, and model_used
    - _Requirements: 8.1, 8.2, 8.5, 8.6, 8.8_
  
  - [x] 7.4 Create FastAPI endpoint for investigation
    - Create POST /investigate endpoint accepting JSON payload
    - Accept question, claim_id (optional), top_k (default 6)
    - Return 200 with investigation report
    - Return 500 on failures with descriptive errors
    - _Requirements: 5.2, 5.3, 5.4, 10.6_
  
  - [ ]* 7.5 Write property test for investigation evidence limit
    - **Property 15: Investigation Evidence Limit**
    - **Validates: Requirements 8.1, 8.6**
  
  - [ ]* 7.6 Write property test for investigation evidence sent to LLM
    - **Property 16: Investigation Evidence Sent to LLM**
    - **Validates: Requirements 8.2**
  
  - [ ]* 7.7 Write property test for claim-scoped investigation
    - **Property 17: Claim-Scoped Investigation**
    - **Validates: Requirements 8.5**
  
  - [ ]* 7.8 Write property test for investigation report includes sources
    - **Property 18: Investigation Report Includes Sources**
    - **Validates: Requirements 8.8**
  
  - [ ]* 7.9 Write unit test for missing evidence file handling
    - Test investigation continues when file not found
    - Verify missing file noted in processing
    - _Requirements: 9.5_

- [ ] 8. Implement error handling and API status codes
  - [x] 8.1 Add global exception handlers to FastAPI app
    - Handle validation errors → 400
    - Handle not found errors → 404
    - Handle internal errors → 500
    - Return consistent error response format
    - _Requirements: 5.3, 5.4, 5.5, 10.4, 10.5, 10.6_
  
  - [ ]* 8.2 Write property test for malformed request error handling
    - **Property 10: Malformed Request Error Handling**
    - **Validates: Requirements 5.3**
  
  - [ ]* 8.3 Write property test for successful request status code
    - **Property 11: Successful Request Status Code**
    - **Validates: Requirements 5.4**
  
  - [ ]* 8.4 Write unit tests for external service error handling
    - Mock Gemini API unavailable
    - Mock ChromaDB unavailable
    - Verify appropriate error messages
    - _Requirements: 10.4, 10.5_

- [x] 9. Checkpoint - Verify backend services
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Scaffold React frontend
  - [x] 10.1 Create React project with TypeScript
    - Initialize React app with Vite or Create React App
    - Set up TypeScript configuration
    - Install dependencies: axios for API calls
    - _Requirements: 7.1, 7.2_
  
  - [x] 10.2 Create API client module
    - Implement `uploadFile()` function
    - Implement `search()` function
    - Implement `investigate()` function
    - Configure base URL for FastAPI backend
    - _Requirements: 5.1, 5.2_
  
  - [x] 10.3 Create FileUpload component
    - Implement drag-and-drop zone
    - Implement click-to-browse file selection
    - Add claim_id and description inputs
    - Add upload progress indicator
    - Display success message with file_id
    - Display error messages on failure
    - _Requirements: 7.1, 7.3, 7.6_
  
  - [x] 10.4 Create SearchBox component
    - Implement text input for queries
    - Add top_k slider (1-20, default 10)
    - Add optional claim_id filter input
    - Add loading indicator during search
    - Display results with metadata and similarity scores
    - Display error messages on failure
    - _Requirements: 7.2, 7.4, 7.5, 7.6_
  
  - [x] 10.5 Create InvestigationPanel component
    - Implement text area for natural language questions
    - Add top_k slider (1-10, default 6)
    - Add optional claim_id filter input
    - Add loading indicator during investigation
    - Display investigation report with formatted answer
    - Display source evidence list with similarity scores
    - Display error messages on failure
    - _Requirements: 7.2, 7.4, 7.5, 7.6_
  
  - [x] 10.6 Create main App component
    - Integrate FileUpload, SearchBox, and InvestigationPanel
    - Add basic styling and layout
    - Add navigation between upload, search, and investigation views
    - _Requirements: 7.1, 7.2_
  
  - [ ]* 10.7 Write component tests for FileUpload
    - Test component renders correctly
    - Test file selection triggers upload
    - Test error display
    - _Requirements: 7.1, 7.3, 7.6_
  
  - [ ]* 10.8 Write component tests for SearchBox
    - Test component renders correctly
    - Test search submission
    - Test results display
    - _Requirements: 7.2, 7.4, 7.5_
  
  - [ ]* 10.9 Write component tests for InvestigationPanel
    - Test component renders correctly
    - Test question submission
    - Test report display
    - _Requirements: 7.2, 7.4, 7.5_

- [ ] 11. Integration and end-to-end testing
  - [ ]* 11.1 Write integration test for complete ingestion flow
    - Upload file → verify embedding stored → verify file on disk
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 9.1_
  
  - [ ]* 11.2 Write integration test for complete search flow
    - Ingest files → search with query → verify relevant results
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [ ]* 11.3 Write integration test for complete investigation flow
    - Ingest evidence → ask question → verify report with sources
    - _Requirements: 8.1, 8.2, 8.8_
  
  - [ ]* 11.4 Write integration test for cross-modal search
    - Ingest video, image, audio, PDF → search → verify cross-modal results
    - _Requirements: 2.6, 4.2_
  
  - [ ]* 11.5 Write integration test for claim-scoped operations
    - Ingest files with different claim_ids → search/investigate with filter → verify scoping
    - _Requirements: 8.5, 9.3_

- [x] 12. Final checkpoint - Complete system verification
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- The backend uses Python with FastAPI, ChromaDB, and Google Gemini APIs
- The frontend uses React with TypeScript
- All property tests should be tagged with: `# Feature: multimodal-ingest, Property {N}: {property_text}`
