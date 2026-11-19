from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional,AsyncGenerator
import asyncio
import json

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

def _init_state(query: str) -> ResearchState:
    return {
        "query": query,
        "search_results": [],
        "num_iterations": 0,
        "critique_notes": "",
        "needs_more_research": True,
        "report": "",
        "past_reports": None,
    }

def _format_sse(event: str, data: Dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.post("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/research/stream")
async def conduct_research_stream(req: ResearchRequest, request: Request):
    async def event_generator() -> AsyncGenerator[str, None]:
        state = _init_state(req.query)

        try:
            yield _format_sse(
                "status",
                {"stage": "memory_lookup", "message": "Retrieving similar past reports..."},
            )

            # Run synchronous ChromaDB + HF embedding lookup off the event loop
            past = await asyncio.to_thread(retrieve_similar_reports, req.query, 3)
            if past:
                state["past_reports"] = past

            # LangGraph automatically runs synchronous node functions in a threadpool
            # and yields {node_name: state_update} as each node completes
            async for update in graph.astream(state, stream_mode="updates"):
                if await request.is_disconnected():
                    break

                for node_name, node_output in update.items():
                    if isinstance(node_output, dict):
                        state.update(node_output)

                    yield _format_sse(
                        "progress",
                        {
                            "node": node_name,
                            "num_iterations": state.get("num_iterations", 0),
                            "needs_more_research": state.get("needs_more_research", False),
                            "critique_notes": state.get("critique_notes", ""),
                        },
                    )

            if not await request.is_disconnected():
                final_payload = ResearchResponse(
                    query=state.get("query", req.query),
                    report=state.get("report", ""),
                    search_results=state.get("search_results", []),
                    critique_notes=state.get("critique_notes", ""),
                    num_iterations=state.get("num_iterations", 0),
                ).model_dump()

                yield _format_sse("complete", final_payload)

        except Exception as exc:
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Prevents proxy/ingress response buffering
        },
    )


@app.post("/research", response_model=ResearchResponse)
async def conduct_research(req: ResearchRequest):
    state = _init_state(req.query)
    
    # state: ResearchState = {
    #     "query": req.query,
    #     "search_results": [],
    #     "num_iterations": 0,
    #     "critique_notes": "",
    #     "needs_more_research": True,
    #     "report": "",
    #     "past_reports": None,
    # }

    # New: retrieve up to 3 similar past research summaries
    past = await asyncio.to_thread(retrieve_similar_reports, req.query, 3)
    #past = retrieve_similar_reports(req.query, k=3)
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
