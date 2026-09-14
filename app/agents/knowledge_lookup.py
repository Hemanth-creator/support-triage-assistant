"""
Agent 2: Knowledge Lookup
Retrieves relevant FAQ context from the FAISS vector store for the customer query.
"""

from app.agents.state import AgentState
from app.rag.retriever import retrieve


def lookup_knowledge(state: AgentState) -> AgentState:
    """
    LangGraph node: reads state['query'], writes state['retrieved_docs'] and state['best_score'].
    """
    query = state["query"]

    results = retrieve(query, k=3)

    state["retrieved_docs"] = results
    # Lower FAISS L2 score = more similar. Take the best (lowest) score as our confidence signal.
    state["best_score"] = results[0]["score"] if results else None

    return state


if __name__ == "__main__":
    test_state: AgentState = {
        "query": "I was charged twice for my subscription this month",
        "intent": "billing",
        "retrieved_docs": None,
        "best_score": None,
        "draft_response": None,
        "escalate": None,
        "escalation_reason": None,
    }
    result = lookup_knowledge(test_state)
    print(f"Best score: {result['best_score']:.4f}\n")
    for doc in result["retrieved_docs"]:
        print(f"- {doc['question']} (score: {doc['score']:.4f})")