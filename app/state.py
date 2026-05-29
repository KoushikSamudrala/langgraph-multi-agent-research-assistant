from typing import TypedDict, Optional, List, Dict


class ResearchState(TypedDict):
    """
    Shared state that flows through the LangGraph.

    This is the single source of truth for:
    - User query
    - Retrieved sources
    - Critique metadata
    - Loop control
    - Final report
    """
    query: str
    search_results: List[Dict]
    critique_notes: Optional[str]
    needs_more_research: bool
    num_iterations: int
    report: Optional[str]
