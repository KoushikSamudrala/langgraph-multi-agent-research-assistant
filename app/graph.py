from langgraph.graph import START, END, StateGraph

from .state import ResearchState
from .agents import search_node, critique_node, synthesis_node


def decide_next_node(state: ResearchState) -> str:
    """
    Routing decision after critique:
    - If critique says we need more research AND
      we haven't hit the iteration cap -> go back to search.
    - Otherwise -> proceed to synthesis.
    """
    needs_more = state.get("needs_more_research", True)
    num_iterations = state.get("num_iterations", 0)
    max_iterations = 3  # can be parameterized later

    if needs_more and num_iterations < max_iterations:
        return "search"
    return "synthesis"


def build_graph():
    """
    Define the LangGraph StateGraph over ResearchState and
    wire nodes + conditional edges for the deep research loop.
    """
    builder = StateGraph(ResearchState)

    builder.add_node("search", search_node)
    builder.add_node("critique", critique_node)
    builder.add_node("synthesis", synthesis_node)

    builder.add_edge(START, "search")
    builder.add_edge("search", "critique")
    builder.add_conditional_edges("critique", decide_next_node)
    builder.add_edge("synthesis", END)

    graph = builder.compile()
    return graph
