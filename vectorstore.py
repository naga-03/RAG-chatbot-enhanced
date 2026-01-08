import os
from typing import List, Dict, cast

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from embeddings import embeddings

# Ensure environment variables from .env are loaded (anchored to project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)

# ---------- CONFIG ----------
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chatbot")
# Default dimension for sentence-transformers/all-MiniLM-L6-v2
DIMENSION = int(os.getenv("EMBED_DIM", "384"))
METRIC = "cosine"

# ---------- SINGLETON ----------
_vectorstore: PineconeVectorStore | None = None


def get_vectorstore() -> PineconeVectorStore:
    """
    Lazily initialize and return a singleton PineconeVectorStore.
    """
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set in environment")

    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        # Wait for index to be ready
        import time
        while True:
            status = pc.describe_index(INDEX_NAME).status
            if status == 'Ready':
                break
            time.sleep(1)

    _vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )

    return _vectorstore


def store_chunks(chunks: List[Document], filename: str) -> None:
    """
    Store document chunks in Pinecone with metadata.
    """
    vectorstore = get_vectorstore()

    for i, chunk in enumerate(chunks):
        metadata = cast(Dict[str, str], chunk.metadata)
        metadata["source"] = filename
        chunk_id = f"{filename}_{i}"
        metadata["chunk_id"] = chunk_id
        chunk.id = chunk_id

    vectorstore.add_documents(chunks)


__all__ = [
    "get_vectorstore",
    "store_chunks",
    "INDEX_NAME",
    "DIMENSION",
    "METRIC",
]
