"""
Central place for paths and settings so nothing is hardcoded inside
the pipeline modules. Every other file should import from here rather
than redefining paths or constants.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"

# Define the parser while ingest the markdown file
# MARKDOWN_PARSER = "none"
MARKDOWN_PARSER = "mistune"

# --- Embedding / vector store settings ----------------------------------
# TODO: decide which embedding model you're using and record it here,
# e.g. "text-embedding-3-small" or a Hugging Face sentence-transformers
# model name. Keep it here (not buried in embedder.py) so it's easy to
# swap and easy to see at a glance what generated your embeddings.
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  

# TODO: pick your vector store (Chroma or Qdrant per the sprint plan)
# and record connection details / collection name here.
VECTOR_STORE_COLLECTION_NAME = "sprint1_corpus"

# --- Chunking settings ---------------------------------------------------

# CHUNKING_STRATEGY = "naive"
# CHUNKING_STRATEGY = "recursive"
CHUNKING_STRATEGY = "structure_aware"
CHUNKING_STRATEGY_VERSION = "v01"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 75

