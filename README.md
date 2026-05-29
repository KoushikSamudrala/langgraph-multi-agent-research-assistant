# LangGraph Multi-Agent Research Assistant

A production-ready multi-agent research pipeline powered by **LangGraph**, **Google Gemini**, **Tavily**, and **ChromaDB**. The system runs a self-critiquing deep research loop, persists compressed research memories, and synthesizes structured reports via a FastAPI backend and a Streamlit frontend.

---

## Architecture

```text
User Query
    │
    ▼
[search_node]  ←──────────────────┐
    │                             │
    ▼                             │  needs_more_research = True
[critique_node]  ─────────────────┘  (up to max_iterations)
    │
    │ needs_more_research = False  OR  num_iterations >= max
    ▼
[synthesis_node] ──▶ save_report_to_memory()
    │
    ▼
 Structured Report
    │
    ▼
Persistent Memory (Chroma: {query, summary} for future runs)
```

### Agents

| Agent | Role |
|---|---|
| **search_node** | Calls Tavily to retrieve up to 5 relevant web sources (title, snippet, url) |
| **critique_node** | Uses Gemini to assess whether results are sufficient; outputs JSON `{needs_more_research, critique_notes}` |
| **synthesis_node** | Uses Gemini to write a 7-section structured report with inline citations and then stores a compressed memory of the run in Chroma |

---

## Project Structure

```text
langgraph-multi-agent-research-assistant/
├── app/
│   ├── __init__.py
│   ├── state.py          # ResearchState TypedDict (query, results, critique, loop control, report, memory)
│   ├── llm.py            # Gemini LLM factory
│   ├── agents.py         # search, critique, synthesis nodes (core agent logic)
│   ├── graph.py          # LangGraph StateGraph wiring + controlled loop
│   ├── api.py            # FastAPI entrypoint exposing /research
│   ├── memory.py         # Chroma + HF embeddings: cross-session research memory
│   └── prompts.py        # Placeholder for prompt templates
├── frontend/
│   ├── __init__.py
│   └── streamlit_app.py  # Streamlit UI calling the FastAPI backend
├── tests/
│   ├── __init__.py
│   └── test_graph_smoke.py
├── .env.example
├── requirements.txt
├── run_api.sh
├── run_ui.sh
└── README.md
```

---

## State model (ResearchState)

All agents and the graph share a single typed state, which makes the workflow easy to reason about and debug:

```python
class ResearchState(TypedDict):
    query: str
    search_results: List[Dict]         # [{title, snippet, url}, ...]
    critique_notes: Optional[str]
    needs_more_research: bool
    num_iterations: int
    report: Optional[str]
    past_reports: Optional[List[Dict]] # [{query, summary}, ...] from Chroma memory
```

- `search_results`: current Tavily hits for this query.
- `critique_notes`: Gemini’s explanation of coverage/gaps.
- `needs_more_research`: bool flag controlling the loop.
- `num_iterations`: how many search–critique cycles have run.
- `report`: final synthesized markdown report.
- `past_reports`: compact memory of similar past research runs (from Chroma).

---

## Memory layer with ChromaDB

### Concept

The memory layer is intentionally **lightweight and high-level**:

- After each successful research run:
  - The full report is compressed into a short summary.
  - `{query, summary}` is stored as a document in a **persistent Chroma collection**.
- For a new query:
  - The system retrieves the top‑k (2–3) most similar past summaries.
  - These are injected into `state["past_reports"]` and surfaced in prompts as “RELATED PAST RESEARCH”.

This gives you **cross-session research memory** without overloading the prompt:
- Only compressed summaries are stored.
- Only a few of the most relevant memories are shown to Gemini.

### Implementation

The memory logic lives in `app/memory.py` and is used in two places:

1. **Write path** – after synthesis:

   - `synthesis_node` (Gemini) generates the full report.
   - It then calls:

     ```python
     save_report_to_memory(query, report)
     ```

   - `save_report_to_memory`:
     - Calls Gemini to produce a compact summary of the report (3–5 key points, ~150–200 words).
     - Uses a local Hugging Face embedding (`sentence-transformers/all-MiniLM-L6-v2`) to embed the summary.
     - Upserts the summary into a persistent Chroma collection:

       - `persist_directory = ./chroma_research`
       - `collection_name = "research_history"`
       - Metadata: `{"query": <original_query>, "created_at": <timestamp>}`

   - Chroma is configured with `persist_directory`, so all memories survive restarts.

2. **Read path** – before graph invocation:

   - In `app/api.py`, the FastAPI `/research` endpoint:
     - Builds the initial `ResearchState`.
     - Calls:

       ```python
       past = retrieve_similar_reports(req.query, k=3)
       if past:
           state["past_reports"] = past
       ```

   - `retrieve_similar_reports`:
     - Uses the same Chroma collection and embedding config.
     - Calls `similarity_search(query, k)` to find the top‑k closest summaries.
     - Maps them into:

       ```python
       [
         {"query": <past_query>, "summary": <short_summary>},
         ...
       ]
       ```

   - These appear as `RELATED PAST RESEARCH` in the synthesis prompt, so Gemini can reuse relevant insights while still prioritizing fresh web results.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/KoushikSamudrala/langgraph-multi-agent-research-assistant.git
cd langgraph-multi-agent-research-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your keys:
# GOOGLE_API_KEY=...
# TAVILY_API_KEY=...
```

Get your keys:

- Gemini: https://aistudio.google.com/app/apikey  
- Tavily: https://app.tavily.com

### 3. Run the backend (FastAPI + LangGraph)

```bash
chmod +x run_api.sh && ./run_api.sh
# or:
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

### 4. Run the frontend (Streamlit)

In a second terminal:

```bash
chmod +x run_ui.sh && ./run_ui.sh
# or:
streamlit run frontend/streamlit_app.py
```

Open: `http://localhost:8501`

---

## API Reference

### `POST /research`

**Request body:**

```json
{
  "query": "What are the latest advances in LangGraph multi-agent systems?",
  "max_iterations": 3
}
```

**Response:**

```json
{
  "query": "...",
  "report": "# Title\n## Executive Summary\n...",
  "search_results": [
    {"title": "...", "snippet": "...", "url": "..."}
  ],
  "critique_notes": "Results are comprehensive and cover all major aspects.",
  "num_iterations": 2
}
```

- `report`: Markdown-style structured report (title, summary, findings, limitations, next steps, sources).
- `search_results`: normalized Tavily results (title, snippet, url).
- `critique_notes`: Gemini’s explanation of whether more research was needed.
- `num_iterations`: how many search–critique loops were executed.

---

## How the loop works (LangGraph)

The `StateGraph` orchestrates the agents over `ResearchState`:

1. **search_node**
   - Reads: `query`
   - Calls Tavily with `max_results=5`, `topic="general"`.
   - Writes: `search_results` as a list of `{title, snippet, url}`.
   - Increments: `num_iterations`.

2. **critique_node**
   - Reads: `query`, `search_results`.
   - Renders results as:

     ```text
      title — snippet (source: url)[1]
      ...[2]
     ```

   - Asks Gemini to return **only JSON**:

     ```json
     {
       "needs_more_research": true/false,
       "critique_notes": "..."
     }
     ```

   - Writes: `needs_more_research`, `critique_notes`.

3. **decide_next_node**
   - Reads: `needs_more_research`, `num_iterations`.
   - If `needs_more_research == True` and `num_iterations < 3` → back to `"search"`.
   - Otherwise → `"synthesis"`.

4. **synthesis_node**
   - Reads: `query`, `search_results`, `critique_notes`, `past_reports`.
   - Builds a structured prompt with:
     - User query.
     - RELATED PAST RESEARCH (if any): up to 3 `{query, summary}` items from Chroma.
     - SEARCH RESULTS: numbered list with sources.
     - Critique notes.
   - Gemini returns a full report with sections:
     1. Title  
     2. Executive Summary  
     3. Background and Problem Context  
     4. Key Findings  
     5. Nuances, Trade-offs, and Limitations  
     6. Recommended Next Steps  
     7. Sources  
   - Writes: `report`.
   - Calls: `save_report_to_memory(query, report)` to persist a compressed memory of the run.

---

## Running Tests

```bash
pytest tests/
```

> **Note:** `test_graph_smoke.py` will make live API calls to Tavily and Gemini. Ensure `.env` is configured and you are okay with small API usage before running.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Google Gemini 1.5 Pro (`langchain-google-genai`) |
| Web search | Tavily |
| Memory / vector store | ChromaDB + local Hugging Face embeddings |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| State schema | Python TypedDict (`ResearchState`) |

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key for Gemini |
| `TAVILY_API_KEY` | Tavily search API key |

---

## License

MIT
