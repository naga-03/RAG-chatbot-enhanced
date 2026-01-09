from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI
from typing import List
import os
import shutil

from chain import get_rag_chain
from dotenv import load_dotenv
from pydantic import BaseModel

from loaders import load_document
from chunking import chunk_documents
from vectorstore import store_chunks
from language import detect_language
from chain import get_rag_chain


class ChatRequest(BaseModel):
    query: str
    session_id: str


app = FastAPI()

# Load environment variables from .env (including PINECONE_API_KEY)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if full_path.startswith(("health", "upload", "chat")):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "frontend", "public", "index.html")
    return FileResponse(index_path)

@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)) -> JSONResponse:
    uploaded_files: List[str] = []
    temp_files: List[str] = []

    try:
        for file in files:
            filename = file.filename or "unknown_file"
            temp_path = f"temp_{filename}"
            temp_files.append(temp_path)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"[UPLOAD] Saved temp file: {temp_path}")

            # Load document
            documents = load_document(temp_path)
            print(f"[UPLOAD] Loaded documents: {len(documents)}")
            if not documents:
                return JSONResponse(status_code=400, content={"error": "Failed to load document"})

            # Chunk documents
            chunks = chunk_documents(documents)
            print(f"[UPLOAD] Generated chunks: {len(chunks)}")
            if not chunks:
                return JSONResponse(status_code=400, content={"error": "No chunks generated"})

            # Store chunks
            store_chunks(chunks, filename)
            print(f"[UPLOAD] Stored chunks in vectorstore")

            uploaded_files.append(filename)

        return JSONResponse(
            content={"message": "Files uploaded and processed", "files": uploaded_files}
        )

    except Exception as e:
        print(f"[UPLOAD] Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    language = detect_language(request.query)

    # Get the RAG chain for the session
    chain = get_rag_chain(request.session_id)

    # Process the query through the chain
    result = chain.invoke({"question": request.query})

    # Extract answer text
    answer_text = result.get("answer", "")
    
    # Extract retrieved chunks properly
    retrieved_chunks = result.get("retrieved_chunks", [])
    serializable_chunks = [chunk["text"] for chunk in retrieved_chunks]

    return JSONResponse(
        content={
            "answer": answer_text,
            "metadata": {
                "language": language,
                "retrieved_chunks": serializable_chunks,
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    import os

    # Use Railway's dynamic port, fallback to 8000 for local testing
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)