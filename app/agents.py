import json
from typing import List, Dict

from tavily import TavilyClient

from .state import ResearchState
from .llm import get_llm

# Tavily client for web research
tavily_client = TavilyClient()


def search_node(state: ResearchState) -> ResearchState:
    """
    Search agent:
    - Reads `query` from state
    - Calls Tavily
    - Normalizes results into {title, snippet, url}
    - Increments `num_iterations`
    """
    query = state.get("query", "").strip()
    if not query:
        # nothing to search – just return state
        return state

    raw = tavily_client.search(
        query=query,
        max_results=5,
        topic="general",
    )

    results: List[Dict] = []
    for item in raw.get("results", []):
        title = item.get("title", "No title")
        snippet = item.get("snippet", item.get("content", "No snippet"))
        url = item.get("url", "")
        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
            }
        )

    state["search_results"] = results
    state["num_iterations"] = state.get("num_iterations", 0) + 1
    return state


def _format_results_for_prompt(search_results: List[Dict]) -> str:
    """
    Render search results in a stable, numbered format so both
    critique and synthesis can reference them with [1], [2], ...
    """
    lines = []
    for i, result in enumerate(search_results, start=1):
        title = result.get("title", "No title")
        snippet = result.get("snippet", "No snippet")
        url = result.get("url", "")
        lines.append(f"[{i}] {title} — {snippet} (source: {url})")
    return "\n".join(lines)


def critique_node(state: ResearchState) -> ResearchState:
    """
    Critique agent (Gemini):
    - Evaluates whether current search_results are sufficient
    - Returns JSON with:
      - needs_more_research: bool
      - critique_notes: str
    - Updates state accordingly
    """
    llm = get_llm()
    query = state.get("query", "")
    search_results = state.get("search_results", [])

    results_text = _format_results_for_prompt(search_results)

    prompt = f"""
You are a research quality critic. Your job is to evaluate whether the current search results are sufficient to answer the user's query, or if more research is needed.

User query:
{query}

SEARCH RESULTS:
{results_text}

Return ONLY a single JSON object with exactly these two keys:
- "needs_more_research": a boolean (true or false)
- "critique_notes": a short string explanation

Do not include any extra keys, comments, markdown, or text outside the JSON.
""".strip()

    response = llm.invoke(prompt)
    raw = str(getattr(response, "content", response))

    data = json.loads(raw)
    state["needs_more_research"] = bool(data.get("needs_more_research", False))
    state["critique_notes"] = str(data.get("critique_notes", ""))

    return state



def synthesis_node(state: ResearchState) -> ResearchState:
    """
    Synthesis agent (Gemini):
    - Reads query, search_results, critique_notes
    - Writes a structured report with sections:
      Title, Executive Summary, Background, Key Findings,
      Nuances/Limitations, Recommended Next Steps, Sources
    - Uses [1], [2], ... citations aligned with SEARCH RESULTS.
    """
    llm = get_llm()
    query = state.get("query", "")
    search_results = state.get("search_results", [])
    critique_notes = state.get("critique_notes", "")

    sources_text = _format_results_for_prompt(search_results)
    past_reports = state.get("past_reports", [])
    if past_reports:
        memory_block_lines = []
        for i, item in enumerate(past_reports, start=1):
            pq = item.get("query", "Unknown query")
            ps = item.get("summary", "")
            memory_block_lines.append(f"[Mem {i}] Query: {pq}\nSummary: {ps}")
        memory_block = "\n\n".join(memory_block_lines)
    else:
        memory_block = "None"
    prompt = f"""
You are a research synthesis assistant. Your task is to synthesize the information from the search results to answer the user's query in a structured report.

User query:
{query}

SEARCH RESULTS:
{sources_text}

CRITIQUE NOTES (may highlight gaps or limitations):
{critique_notes}
RELATED PAST RESEARCH (if any):
{memory_block}

Write the report with the following sections, in this exact order:

1. Title
2. Executive Summary
3. Background and Problem Context
4. Key Findings
5. Nuances, Trade-offs, and Limitations
6. Recommended Next Steps
7. Sources

Guidelines:
- Use citation markers like [1], [2], [3] in the text whenever you rely on a specific source.
- The numbers must correspond to the numbered SEARCH RESULTS above.
- Be concise but specific. Avoid generic statements that are not grounded in the sources. 
- You may reuse insights from RELATED PAST RESEARCH if they are clearly relevant, but always prioritize up-to-date web results when there is a conflict.
""".strip()

    response = llm.invoke(prompt)
    report_text = str(getattr(response, "content", response))

    state["report"] = report_text
    # New: persist this run as a memory
    query = state.get("query", "")
    if query and report_text:
        save_report_to_memory(query, report_text)
    return state
