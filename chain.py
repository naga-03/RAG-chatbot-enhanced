from typing import List, Dict, Any
import os

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq

from prompt import SYSTEM_PROMPT, USER_PROMPT
from session_memory import SessionMemory
from vectorstore import get_vectorstore

# -------------------- Session Store --------------------
session_memories: Dict[str, SessionMemory] = {}

def get_session_memory(session_id: str) -> SessionMemory:
    if session_id not in session_memories:
        session_memories[session_id] = SessionMemory()
    return session_memories[session_id]

# -------------------- LLM Selector --------------------
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return ChatGroq(model=model)
    return Ollama(model=os.getenv("OLLAMA_MODEL", "llama2"))

# -------------------- RAG Chain --------------------
def get_rag_chain(session_id: str) -> RunnableLambda[Dict[str, str], Dict[str, Any]]:
    """
    Returns a RunnableLambda that takes {"question": str} and returns:
        {
            "answer": str,
            "retrieved_chunks": List[Dict[str, Any]]
        }
    """
    vectorstore = get_vectorstore()
    llm = get_llm()

    def chat_flow(input_dict: Dict[str, str]) -> Dict[str, Any]:
        question = input_dict.get("question", "")
        memory = get_session_memory(session_id)

        # -------------------- Retrieve relevant chunks --------------------
        docs_with_scores = vectorstore.similarity_search_with_score(question, k=4)

        docs: List[Document] = [doc for doc, score in docs_with_scores]

        # Format context for LLM
        context = "\n\n".join(doc.page_content for doc in docs)

        # -------------------- Build Prompt --------------------
        prompt = f"""
{SYSTEM_PROMPT}

{USER_PROMPT.format(context=context, question=question)}
"""

        # -------------------- Call LLM --------------------
        answer = llm.invoke(prompt)
        answer_text = getattr(answer, "content", str(answer))

        # -------------------- Store in Session Memory --------------------
        memory.add_turn(question, answer_text)

        # -------------------- Prepare retrieved chunks --------------------
        retrieved_chunks = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "similarity_score": score
            }
            for doc, score in docs_with_scores
        ]

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks
        }

    return RunnableLambda(chat_flow)
