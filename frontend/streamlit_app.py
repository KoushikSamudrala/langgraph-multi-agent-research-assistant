import streamlit as st
import requests

API_URL = "http://localhost:8000/research"


st.title("LangGraph Multi-Agent Research Assistant")
st.markdown(
    "This app uses a LangGraph-powered multi-agent pipeline "
    "to run deep web research with Tavily and synthesize a structured report with Gemini."
)

query = st.text_area(
    "Your research question:",
    placeholder="e.g., LangGraph for multi-agent orchestration: how it works, when to use it, and advantages over plain LangChain.",
)

max_iterations = st.slider(
    "Max research iterations (depth)",
    min_value=1,
    max_value=5,
    value=3,
)

run_clicked = st.button("Run deep research")

if run_clicked:
    if not query.strip():
        st.error("Please enter a research question before running.")
    else:
        with st.spinner("Conducting deep research..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query, "max_iterations": max_iterations},
                    timeout=120,
                )
            except Exception as e:
                st.error(f"Request failed: {e}")
            else:
                if response.status_code == 200:
                    result = response.json()

                    st.subheader("Research Report")
                    st.markdown(result.get("report", "No report generated."))

                    st.subheader("Sources")
                    for idx, res in enumerate(result.get("search_results", []), start=1):
                        title = res.get("title", f"Source {idx}")
                        url = res.get("url", "")
                        snippet = res.get("snippet", "")
                        if url:
                            st.markdown(f"[{idx}] [{title}]({url})")
                        else:
                            st.markdown(f"[{idx}] {title}")
                        if snippet:
                            st.caption(snippet)

                    st.subheader("Run metadata")
                    st.write("Iterations used:", result.get("num_iterations", 0))
                    st.write("Critique notes:", result.get("critique_notes", ""))
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
