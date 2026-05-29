from typing import List, Dict

def save_report_to_memory(query: str, report: str) -> None:
    """
    - Summarize the report into a short memory summary (LLM or simple heuristic).
    - Upsert into Chroma with `query` as metadata.
    """
    ...

def retrieve_similar_reports(query: str, k: int = 3) -> List[Dict]:
    """
    - Use Chroma similarity search on summaries.
    - Return up to k items like:
      [{"query": "...", "summary": "..."}, ...]
    """
    ...
