"""
Agent 1: Intent Classifier
Classifies the customer's query into a support category using Groq's LLM.
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from app.agents.state import AgentState

load_dotenv()

INTENT_CATEGORIES = [
    "billing",
    "account",
    "technical",
    "refund",
    "general_inquiry",
]

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


def classify_intent(state: AgentState) -> AgentState:
    """
    LangGraph node: reads state['query'], writes state['intent'].
    """
    query = state["query"]

    prompt = f"""Classify the following customer support query into exactly ONE of these categories:
{', '.join(INTENT_CATEGORIES)}

Query: "{query}"

Respond with ONLY the category name, nothing else."""

    response = llm.invoke(prompt)
    intent = response.content.strip().lower()

    # Fallback safety: if the model returns something unexpected, default to general_inquiry
    if intent not in INTENT_CATEGORIES:
        intent = "general_inquiry"

    state["intent"] = intent
    return state


if __name__ == "__main__":
    test_state: AgentState = {
        "query": "I was charged twice for my subscription this month",
        "intent": None,
        "retrieved_docs": None,
        "best_score": None,
        "draft_response": None,
        "escalate": None,
        "escalation_reason": None,
    }
    result = classify_intent(test_state)
    print("Classified intent:", result["intent"])