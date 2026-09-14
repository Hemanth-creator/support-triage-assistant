"""
Builds the LangGraph StateGraph wiring together:
intent_classifier -> knowledge_lookup -> response_drafter -> END
"""

from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.intent_classifier import classify_intent
from app.agents.knowledge_lookup import lookup_knowledge
from app.agents.response_drafter import draft_response


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("lookup_knowledge", lookup_knowledge)
    graph.add_node("draft_response", draft_response)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "lookup_knowledge")
    graph.add_edge("lookup_knowledge", "draft_response")
    graph.add_edge("draft_response", END)

    return graph.compile()


# Compiled once, reused across requests (important for FastAPI later)
compiled_graph = build_graph()


def run_pipeline(query: str) -> dict:
    """
    Runs the full multi-agent pipeline for a given customer query.
    """
    initial_state: AgentState = {
        "query": query,
        "intent": None,
        "retrieved_docs": None,
        "best_score": None,
        "draft_response": None,
        "escalate": None,
        "escalation_reason": None,
    }
    final_state = compiled_graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    test_queries = [
        "I want a refund for my last purchase",
        "How do I train my dog to sit?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = run_pipeline(q)
        print(f"Intent: {result['intent']}")
        print(f"Best score: {result['best_score']}")
        print(f"Escalate: {result['escalate']}")
        print(f"Reason: {result['escalation_reason']}")
        print(f"Response: {result['draft_response']}")