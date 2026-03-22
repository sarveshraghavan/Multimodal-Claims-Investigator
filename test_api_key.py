import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "")
print(f"API Key loaded: {bool(api_key)}")
print(f"API Key (first 20 chars): {api_key[:20] if api_key else 'None'}...")
print(f"API Key length: {len(api_key) if api_key else 0}")
