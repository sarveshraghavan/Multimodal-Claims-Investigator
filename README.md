# 🔍 Multimodal Claims Investigator

An AI-powered evidence analysis system that enables semantic search and intelligent investigation across multiple file types (video, image, audio, PDF) using Google's Gemini API and vector embeddings.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![React](https://img.shields.io/badge/React-18+-61DAFB.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

### 🎯 Core Capabilities
- **Multimodal File Upload**: Support for video (.mp4, .mov), images (.jpg, .png), audio (.mp3, .wav), and documents (.pdf)
- **Semantic Search**: Natural language search across all file types using vector embeddings
- **AI Investigation**: Ask complex questions and get AI-generated analysis reports using Gemini 2.5 Flash
- **Cross-Modal Analysis**: Search and analyze evidence across different modalities simultaneously
- **Claim Management**: Organize files by claim ID for structured investigation workflows

### 🎨 Modern UI
- Beautiful gradient-themed interface with smooth animations
- Drag & drop file upload with progress tracking
- Real-time search results with similarity scores
- Typing animation for AI-generated reports
- Color-coded file type badges and relevance indicators
- Responsive design for desktop and mobile

### 🔧 Technical Features
- Vector database storage with ChromaDB
- Multimodal embeddings using Gemini API
- RESTful API with FastAPI
- Comprehensive test suite (96+ tests)
- Type-safe frontend with TypeScript
- Modular, maintainable architecture

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/multimodal-claims-investigator.git
cd multimodal-claims-investigator
```

2. **Set up the backend**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

3. **Set up the frontend**
```bash
cd frontend
npm install
cd ..
```

4. **Start the servers**

Backend (Terminal 1):
```bash
python run.py
```

Frontend (Terminal 2):
```bash
cd frontend
npm run dev
```

5. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📖 Usage

### Upload Evidence
1. Navigate to the **Upload** tab
2. Drag & drop files or click to browse
3. Optionally add a claim ID and description
4. Click "Upload File"

### Search Evidence
1. Go to the **Search** tab
2. Enter a natural language query (e.g., "dashcam footage showing collision")
3. Optionally filter by claim ID
4. Adjust max results slider
5. View results with similarity scores

### AI Investigation
1. Navigate to the **Investigate** tab
2. Enter your question (e.g., "Does the driver's statement match the dashcam footage?")
3. Optionally filter by claim ID
4. Set number of evidence files to analyze
5. Click "Start Investigation"
6. Watch the AI generate a detailed report with source citations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + TS)                    │
│  ┌──────────┐  ┌──────────┐  ┌─────────────────────┐       │
│  │  Upload  │  │  Search  │  │   Investigation     │       │
│  └──────────┘  └──────────┘  └─────────────────────┘       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
┌────────────────────────▼────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Ingest  │  │  Embed   │  │  Search  │  │ Retrieve │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
┌───────▼────────┐              ┌─────────▼──────────┐
│   ChromaDB     │              │   Gemini API       │
│ (Vector Store) │              │ (Embeddings + LLM) │
└────────────────┘              └────────────────────┘
```

## 📁 Project Structure

```
multimodal-claims-investigator/
├── app/                      # Backend application
│   ├── main.py              # FastAPI app entry point
│   ├── ingest.py            # File upload & ingestion
│   ├── embed.py             # Multimodal embedding generation
│   ├── search.py            # Semantic search
│   ├── retrieval.py         # AI investigation
│   └── db.py                # ChromaDB operations
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── App.tsx          # Main app component
│   │   ├── FileUpload.tsx   # Upload interface
│   │   ├── SearchBox.tsx    # Search interface
│   │   ├── InvestigationPanel.tsx  # Investigation interface
│   │   └── api.ts           # API client
│   └── package.json
├── tests/                    # Comprehensive test suite
├── uploads/                  # Uploaded files storage
├── chroma_data/             # Vector database storage
├── requirements.txt         # Python dependencies
├── run.py                   # Backend startup script
└── README.md
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_search.py
```

Test coverage includes:
- File upload and ingestion
- Embedding generation for all modalities
- Semantic search functionality
- AI investigation endpoints
- Database operations
- Error handling

## 🔑 API Endpoints

### Upload
```http
POST /upload
Content-Type: multipart/form-data

file: <file>
claim_id: <optional>
description: <optional>
```

### Search
```http
POST /search
Content-Type: application/json

{
  "query": "dashcam footage",
  "top_k": 10,
  "claim_id": "CLM-2024-001"
}
```

### Investigate
```http
POST /investigate
Content-Type: application/json

{
  "question": "What happened in the collision?",
  "claim_id": "CLM-2024-001",
  "top_k": 6
}
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **ChromaDB**: Vector database for embedding storage and similarity search
- **Google Gemini API**: Multimodal embeddings and LLM capabilities
- **Python 3.8+**: Core programming language
- **Pytest**: Testing framework

### Frontend
- **React 18**: UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **Axios**: HTTP client for API calls

### AI/ML
- **Gemini Embedding Model**: `gemini-embedding-001` for multimodal embeddings
- **Gemini LLM**: `gemini-2.5-flash` for investigation reports
- **Vector Search**: Cosine similarity for semantic matching

## 🎯 Use Cases

### Insurance Claims Investigation
- Upload accident photos, dashcam videos, witness statements (audio), and police reports (PDF)
- Search for specific evidence types across all files
- Ask AI to analyze consistency between different evidence sources

### Legal Discovery
- Organize evidence by case number
- Perform semantic search across diverse document types
- Generate AI-powered analysis reports for case review

### Content Management
- Store and search multimedia content
- Find related content across different formats
- AI-assisted content analysis and summarization

## 🔒 Security & Privacy

- API keys stored in environment variables
- Files stored locally (not sent to external services except for embedding generation)
- No data persistence beyond local storage
- CORS configured for local development

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for multimodal AI capabilities
- ChromaDB for vector database functionality
- FastAPI and React communities for excellent documentation

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/multimodal-claims-investigator](https://github.com/yourusername/multimodal-claims-investigator)

---

⭐ If you find this project useful, please consider giving it a star!
