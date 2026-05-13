import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # used for embeddings only

CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "default"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "claude-sonnet-4-5"  # adjust to current Sonnet model id