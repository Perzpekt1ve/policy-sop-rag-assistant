from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
ASSETS_DIR = BASE_DIR / "assets"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:1.5b")
HF_EMBED_MODEL = os.getenv(
    "HF_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CHUNK_SIZE = 200
CHUNK_OVERLAP = 100
RETRIEVAL_K = 4
MAX_SOURCE_PREVIEW_CHARS = 700

APP_TITLE = "Policy / SOP Assistant"
APP_SUBTITLE = "Ask grounded questions over uploaded policies, SOPs, manuals, and handbooks."