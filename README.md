# LangGraph Multi-Agent Research Assistant

A production-ready multi-agent research pipeline powered by **LangGraph**, **Google Gemini**, and **Tavily**. The system runs a self-critiquing deep research loop and synthesizes structured reports via a FastAPI backend and a Streamlit frontend.

---

## Architecture

```
User Query
    │
    ▼
[search_node]  ←──────────────────┐
    │                             │
    ▼                             │ needs_more_research = True
[critique_node]  ─────────────────┘  (up to max_iterations)
    │
    │ needs_more_research = False  OR  num_iterations >= max
    ▼
[synthesis_node]
    │
    ▼
 Structured Report
```

### Agents

| Agent | Role |
|---|---|
| **search_node** | Calls Tavily to retrieve up to 5 relevant web sources |
| **critique_node** | Uses Gemini to assess whether results are sufficient; outputs JSON `{needs_more_research, critique_notes}` |
| **synthesis_node** | Uses Gemini to write a 7-section structured report with inline citations |

---

## Project Structure

```
langgraph-multi-agent-research-assistant/
├── app/
│   ├── __init__.py
│   ├── state.py          # ResearchState TypedDict
│   ├── llm.py            # Gemini LLM factory
│   ├── agents.py         # search, critique, synthesis nodes
│   ├── graph.py          # LangGraph StateGraph wiring
│   ├── api.py            # FastAPI entrypoint
│   └── prompts.py        # Placeholder for prompt templates
├── frontend/
│   ├── __init__.py
│   └── streamlit_app.py  # Streamlit UI
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
- Gemini: [Google AI Studio](https://aistudio.google.com/app/apikey)
- Tavily: [Tavily Dashboard](https://app.tavily.com)

### 3. Run the backend

```bash
chmod +x run_api.sh && ./run_api.sh
# or: uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: `http://localhost:8000/docs`

### 4. Run the frontend

In a second terminal:

```bash
chmod +x run_ui.sh && ./run_ui.sh
# or: streamlit run frontend/streamlit_app.py
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
  "search_results": [{"title": "...", "snippet": "...", "url": "..."}],
  "critique_notes": "Results are comprehensive and cover all major aspects.",
  "num_iterations": 2
}
```

---

## Running Tests

```bash
pytest tests/
```

> **Note:** `test_graph_smoke.py` will make live API calls to Tavily and Gemini. Ensure `.env` is configured before running.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Google Gemini 1.5 Pro (via `langchain-google-genai`) |
| Web search | Tavily |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| State schema | Python TypedDict |

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key for Gemini |
| `TAVILY_API_KEY` | Tavily search API key |

---

## License

MIT
