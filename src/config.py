"""Central configuration for the local RAG system."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"

# Ensure dirs exist
for d in (DATA_DIR, MODELS_DIR, CHROMA_DIR, SCREENSHOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "synthetic_corpus.json"

# Model config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_URL = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf"
LLM_MODEL_PATH = MODELS_DIR / "gemma-2-2b-it-Q4_K_M.gguf"

# Generation hyperparameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512
DEFAULT_RETRIEVAL_K = 4

# Chunking
CHUNK_SIZE = 300       # approximate tokens (using character proxy)
CHUNK_OVERLAP = 50

# Streamlit
STREAMLIT_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
