from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.
    chunk_size=1000, chunk_overlap=200
    Language-independent, preserves context across chunks.
    """
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks: List[Document] = text_splitter.split_documents(documents)
    return chunks
