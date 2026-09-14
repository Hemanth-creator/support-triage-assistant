"""
Shared state object passed between all agents in the LangGraph pipeline.
"""

from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    query: str                          # original customer query
    intent: Optional[str]               # classified intent (billing, account, technical, etc.)
    retrieved_docs: Optional[List[dict]]  # top-k FAQ matches from FAISS
    best_score: Optional[float]         # lowest (best) similarity score from retrieval
    draft_response: Optional[str]       # the drafted response text
    escalate: Optional[bool]            # whether this needs a human
    escalation_reason: Optional[str]    # why it was escalated, if applicable