from app.graph import build_graph
from app.state import ResearchState


def test_graph_smoke():
    graph = build_graph()
    state: ResearchState = {
        "query": "What is LangGraph?",
        "search_results": [],
        "num_iterations": 0,
        "critique_notes": "",
        "needs_more_research": True,
        "report": "",
    }
    # This just ensures the graph can be invoked without raising
    final_state = graph.invoke(state)
    assert "report" in final_state
