import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# Models
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
IMAGE_MODEL = "gpt-image-1"

# Data & Chroma
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "book_summaries.json"
CHROMA_DIR = str(Path(__file__).resolve().parents[1] / "chroma_db")
COLLECTION_NAME = "book_summaries"

# Backend settings
DEFAULT_TOP_K = 4
BAD_WORDS = {"prost", "idiot", "jignire", "urât", "hateword", "urat", "stupid"}  
#MAX_TOKENS = 4096  # model context length