from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

from .state import ResearchState
from .graph import build_graph
from .memory import retrieve_similar_reports


app = FastAPI(title="LangGraph Multi-Agent Research Assistant")

# Build the LangGraph once at startup
graph = build_graph()


class ResearchRequest(BaseModel):
    query: str
    max_iterations: int = 3  # reserved for future use


class ResearchResponse(BaseModel):
    query: str
    report: str
    search_results: List[Dict]
    critique_notes: Optional[str]
    num_iterations: int


@app.post("/research", response_model=ResearchResponse)
async def conduct_research(req: ResearchRequest):
    state: ResearchState = {
        "query": req.query,
        "search_results": [],
        "num_iterations": 0,
        "critique_notes": "",
        "needs_more_research": True,
        "report": "",
        "past_reports": None,
    }

    # New: retrieve up to 3 similar past research summaries
    past = retrieve_similar_reports(req.query, k=3)
    if past:
        state["past_reports"] = past

    final_state = graph.invoke(state)

    return ResearchResponse(
        query=final_state.get("query", req.query),
        report=final_state.get("report", ""),
        search_results=final_state.get("search_results", []),
        critique_notes=final_state.get("critique_notes", ""),
        num_iterations=final_state.get("num_iterations", 0),
    )
