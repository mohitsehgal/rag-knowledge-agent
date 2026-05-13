import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "default"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"