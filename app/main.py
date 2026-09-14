"""
FastAPI service exposing the customer support triage pipeline.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.graph import run_pipeline

app = FastAPI(
    title="Customer Support Triage & Response Assistant",
    description="Multi-agent LLM pipeline for classifying, retrieving context for, and drafting responses to customer support queries.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    intent: str | None
    draft_response: str | None
    escalate: bool | None
    escalation_reason: str | None
    best_score: float | None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def submit_query(request: QueryRequest):
    """
    Submit a customer query and receive a routed, drafted response.
    """
    result = run_pipeline(request.query)

    return QueryResponse(
        query=request.query,
        intent=result.get("intent"),
        draft_response=result.get("draft_response"),
        escalate=result.get("escalate"),
        escalation_reason=result.get("escalation_reason"),
        best_score=result.get("best_score"),
    )