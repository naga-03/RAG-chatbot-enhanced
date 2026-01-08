import os
from langchain_community.embeddings import HuggingFaceEmbeddings

# Local / HF embeddings; does NOT use OpenAI.
# Default: sentence-transformers/all-MiniLM-L6-v2 (dim=384)
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
