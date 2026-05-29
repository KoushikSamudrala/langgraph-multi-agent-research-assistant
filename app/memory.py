from typing import List, Dict
import os
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .llm import get_llm 

PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_research")
COLLECTION_NAME = "research_history"

# Initialize embedding model once (local; downloaded on first use)
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Initialize / load persistent Chroma collection
_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR,
    embedding_function=_embeddings,
)


def _summarize_report(report: str) -> str:
    """
    Use Gemini to produce a compact memory summary of the report.
    This keeps memory small while preserving key insights.

    You can simplify this later or change the style.
    """
    llm = get_llm()
    prompt = f"""
You are a memory compression assistant.

Given the following research report, produce a short summary that captures
the main topic and 3–5 key points. The summary should be at most 150–200 words.

Report:
{report}
""".strip()

    response = llm.invoke(prompt)
    summary = str(getattr(response, "content", response))
    return summary.strip()


def save_report_to_memory(query: str, report: str) -> None:
    """
    Persist the finished research run into Chroma as a memory entry.

    - Summarize the full report into a compact summary.
    - Upsert into a persistent Chroma collection with metadata:
      { "query": <original_query>, "created_at": <timestamp> }.
    """
    query = (query or "").strip()
    report = (report or "").strip()
    if not query or not report:
        return None

    # 1. Summarize the report
    summary = _summarize_report(report)
    if not summary:
        return None

    # 2. Upsert into Chroma as a new document
    metadata: Dict = {
        "query": query,
        "created_at": datetime.utcnow().isoformat(),
    }

    _vectorstore.add_texts(texts=[summary], metadatas=[metadata])
    _vectorstore.persist()
    return None

def retrieve_similar_reports(query: str, k: int = 3) -> List[Dict]:
    """
    Retrieve up to k most similar past research summaries for a new query.

    Returns a list of dicts:
    [
      {"query": "<past_query>", "summary": "<short_summary>"},
      ...
    ]
    """
    query = (query or "").strip()
    if not query:
        return []

    # If vectorstore is empty, Chroma will just return an empty list
    docs = _vectorstore.similarity_search(query, k=k)

    memories: List[Dict] = []
    for doc in docs:
        past_query = doc.metadata.get("query", "Unknown query")
        summary = doc.page_content
        memories.append(
            {
                "query": past_query,
                "summary": summary,
            }
        )
    return memories
