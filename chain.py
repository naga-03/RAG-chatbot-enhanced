from typing import List, Dict
import os

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

from prompt import SYSTEM_PROMPT, USER_PROMPT
from session_memory import SessionMemory
from vectorstore import get_vectorstore


# Global store for session memories
session_memories: Dict[str, SessionMemory] = {}


def get_session_memory(session_id: str) -> SessionMemory:
    """
    Get or create a SessionMemory for the given session_id.
    """
    if session_id not in session_memories:
        session_memories[session_id] = SessionMemory()
    return session_memories[session_id]


def _get_llm():
    """
    Select LLM based on environment:
    - LLM_PROVIDER=groq -> Groq Chat model (requires GROQ_API_KEY)
    - otherwise uses local Ollama
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ChatGroq(model=model)

    # Default: Ollama
    ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
    return Ollama(model=ollama_model)


def get_rag_chain(session_id: str) -> RunnableLambda:
    """
    Assemble and return the RAG chain for the given session_id using LCEL.
    Components: Retriever, Prompt, LLM, Memory.
    """
    # Vectorstore for similarity search with scores
    vectorstore = get_vectorstore()

    # LLM (Ollama or Groq, depending on env)
    llm = _get_llm()

    # Chain using LCEL
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def chat_flow(input_dict: dict[str, str]):
        question = input_dict.get("question", "")
        session_memory = get_session_memory(session_id)

        # Fetch chat history
        chat_history = session_memory.get_formatted_history()

        # Create enhanced query
        enhanced_query = f"Conversation so far:\n{chat_history}\n\nUser question:\n{question}"

        # Retrieve documents with scores
        docs_with_scores = vectorstore.similarity_search_with_score(enhanced_query, k=4)

        # Extract docs and add similarity scores to metadata
        docs = []
        retrieved_chunks = []
        for doc, score in docs_with_scores:
            doc.metadata["similarity_score"] = score
            docs.append(doc)
            retrieved_chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "similarity_score": score
            })

        # Combine into context string
        context = format_docs(docs)

        # Create prompt
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(context=context, question=question)}"

        # Stream LLM response
        full_response = ""
        for chunk in llm.stream(prompt):
            chunk_text = str(chunk)
            full_response += chunk_text
            yield chunk_text

        # Store in SessionMemory after streaming completes
        session_memory.add_turn(question, full_response)

    return RunnableLambda(chat_flow)
