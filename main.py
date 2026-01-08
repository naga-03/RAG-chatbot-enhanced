from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List
import os
import shutil

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

@app.get("/")
async def serve_frontend():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "frontend", "index.html")
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

            documents = load_document(temp_path)
            chunks = chunk_documents(documents)
            store_chunks(chunks, filename)

            uploaded_files.append(filename)

        return JSONResponse(
            content={"message": "Files uploaded and processed", "files": uploaded_files}
        )

    except ValueError as e:
        # Unsupported file type
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {str(e)}"}
        )
    except RuntimeError as e:
        # Missing environment variables or Pinecone issues
        return JSONResponse(
            status_code=500,
            content={"error": f"Server configuration error: {str(e)}"}
        )
    except Exception as e:
        # General error
        return JSONResponse(
            status_code=500,
            content={"error": f"Upload failed: {str(e)}"}
        )
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass  # Ignore cleanup errors

@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    language = detect_language(request.query)

    # Get the RAG chain for the session
    chain = get_rag_chain(request.session_id)

    # Process the query through the chain
    result = chain.invoke({"question": request.query})

    return JSONResponse(
        content={
            "answer": result["answer"],
            "metadata": {
                "language": language,
                "retrieved_chunks": result["retrieved_chunks"],
            }
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)