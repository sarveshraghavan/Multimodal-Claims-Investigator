# Multimodal Claims Investigator - Quick Start Guide

## 🚀 Getting Started

### 1. Start the Servers

**Backend (Port 8000):**
```bash
python run.py
```

**Frontend (Port 3001):**
```bash
cd frontend
npm run dev
```

### 2. Access the Application

- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📋 Features Overview

### 📤 Upload Tab
Upload evidence files (video, image, audio, PDF) with:
- Drag & drop interface with animations
- File type icons and size display
- Progress bar during upload
- Optional claim ID and description
- Real-time upload feedback

### 🔍 Search Tab
Semantic search across all uploaded files:
- Natural language queries
- Filter by claim ID
- Adjustable result count (1-20)
- Color-coded modality badges
- Similarity score indicators
- Animated result cards with hover effects

### 🔬 Investigate Tab
AI-powered investigation using Gemini 2.5 Flash:
- Ask complex questions about evidence
- Analyze multiple files simultaneously
- Typing animation for AI responses
- Evidence source cards with relevance scores
- Cross-modal analysis capabilities

## 🎨 UI Features

### Modern Design
- Purple gradient theme
- Smooth animations and transitions
- Card-based layout
- Responsive design
- Glass-morphism effects

### Interactive Elements
- Hover effects on all interactive components
- Progress indicators
- Loading spinners
- Animated result displays
- Typing effect for AI responses

### Visual Feedback
- Color-coded file types (video, image, audio, document)
- Similarity/relevance score badges
- Success/error messages with animations
- File counter in header

## 📝 Example Workflows

### Workflow 1: Upload and Search
1. Go to Upload tab
2. Drag & drop a dashcam video
3. Add claim ID: "CLM-2024-001"
4. Upload the file
5. Go to Search tab
6. Search: "collision footage"
7. View results with similarity scores

### Workflow 2: AI Investigation
1. Upload multiple evidence files (video, audio, images)
2. Go to Investigate tab
3. Ask: "Does the driver's statement match the dashcam footage?"
4. Set evidence files to analyze: 6
5. Watch AI generate report with typing animation
6. Review evidence sources used in analysis

## 🔧 Configuration

### Environment Variables (.env)
```
GEMINI_API_KEY=your_api_key_here
```

### Available Models
- Embedding: `gemini-embedding-001`
- Investigation: `gemini-2.5-flash`

## 📊 System Architecture

```
Frontend (React + TypeScript + Vite)
    ↓
FastAPI Backend
    ↓
ChromaDB (Vector Storage) + Gemini API (Embeddings + LLM)
```

## 🎯 Key Technologies

- **Frontend**: React, TypeScript, Vite
- **Backend**: FastAPI, Python
- **Database**: ChromaDB (vector database)
- **AI**: Google Gemini API
- **Embeddings**: Multimodal embeddings (text, image, video, audio)

## 💡 Tips

1. **Upload files with claim IDs** for better organization
2. **Use descriptive search queries** for better results
3. **Adjust the number of evidence files** in investigations based on your needs
4. **Try cross-modal searches** (e.g., search for "collision" to find videos, images, and audio)
5. **Ask specific questions** in investigations for more focused AI analysis

## 🐛 Troubleshooting

### Port Already in Use
If port 3000 is in use, Vite will automatically use 3001 or the next available port.

### API Connection Errors
- Ensure backend is running on port 8000
- Check that GEMINI_API_KEY is set in .env file
- Verify API key is valid

### Empty Search Results
- Make sure files are uploaded successfully
- Check that embeddings were generated (no errors in backend logs)
- Try uploading files again if database is empty

## 📚 Additional Resources

- **Project Explanation**: See `PROJECT_EXPLANATION.md` for detailed technical documentation
- **API Documentation**: Visit http://localhost:8000/docs when backend is running
- **Test Files**: Use `create_test_files.py` to generate sample files for testing
