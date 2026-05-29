import os
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    """
    Central Gemini configuration.

    This keeps model selection and temperature in one place so the
    rest of the system just calls `get_llm()` and doesn't worry
    about API keys or provider details.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    llm = ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-1.5-pro",
        temperature=0.1,  # low temperature for research consistency
    )
    return llm
