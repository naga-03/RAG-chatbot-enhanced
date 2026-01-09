import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from embeddings import embeddings

# ---------- LOAD ENV ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-chatbot-hf")
DIMENSION = int(os.getenv("EMBED_DIM", "384"))
METRIC = "cosine"

# ---------- INIT PINECONE CLIENT ----------
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("PINECONE_API_KEY not set in environment")

pc = Pinecone(api_key=api_key)

# Create index if it doesn't exist
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east1")  # match your previous specs
    )

# ---------- SINGLETON VECTORSTORE ----------
_vectorstore: PineconeVectorStore | None = None

def get_vectorstore() -> PineconeVectorStore:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    _vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    return _vectorstore

def store_chunks(chunks: List[Document], filename: str) -> None:
    vectorstore = get_vectorstore()
    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}_{i}"
        chunk.id = chunk_id
        if chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata["source"] = filename
        chunk.metadata["chunk_id"] = chunk_id

    vectorstore.add_documents(chunks)

def search_query(query: str, k: int = 4) -> List[Document]:
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)

__all__ = ["get_vectorstore", "store_chunks", "search_query", "INDEX_NAME", "DIMENSION", "METRIC"]
