import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "")
print(f"API Key: {api_key[:20]}...")

genai.configure(api_key=api_key)

# Test text embedding
try:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content="test"
    )
    print(f"✓ Text embedding works! Dimension: {len(result['embedding'])}")
except Exception as e:
    print(f"✗ Text embedding failed: {e}")

# Test with gemini-embedding-001
try:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content="test",
        task_type="retrieval_document"
    )
    print(f"✓ Gemini embedding-001 works! Dimension: {len(result['embedding'])}")
except Exception as e:
    print(f"✗ Gemini embedding-001 failed: {e}")
