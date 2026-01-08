from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from typing import List
import os

def load_document(file_path: str) -> List[Document]:
    """
    Load document based on file type.
    Supports PDF, DOCX, and TXT files.
    Preserves multilingual content.
    """
    file_extension: str = os.path.splitext(file_path)[1].lower()

    loader: BaseLoader
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents: List[Document] = loader.load()
    return documents
