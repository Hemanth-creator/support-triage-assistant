"""
Agent 3: Response Drafter
Drafts a grounded response using retrieved FAQ context, and decides
whether to escalate using a hybrid signal: FAISS similarity score + LLM self-assessment.
"""

import os
import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from app.agents.state import AgentState

load_dotenv()

# Empirically tuned threshold: above this L2 distance, retrieval is considered weak.
SCORE_ESCALATION_THRESHOLD = 1.0

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY"),
)


def draft_response(state: AgentState) -> AgentState:
    """
    LangGraph node: reads query/intent/retrieved_docs, writes draft_response/escalate/escalation_reason.
    """
    query = state["query"]
    intent = state.get("intent", "general_inquiry")
    retrieved_docs = state.get("retrieved_docs") or []
    best_score = state.get("best_score")

    context_block = "\n\n".join(
        f"Q: {d['question']}\nA: {d['answer']}" for d in retrieved_docs
    ) or "No relevant documentation found."

    prompt = f"""You are a customer support assistant. A customer submitted this query:

"{query}"

Classified intent: {intent}

Here is relevant knowledge base context:
{context_block}

Instructions:
1. Draft a helpful, concise response to the customer USING ONLY the information in the context above. Do not invent policies, dates, or numbers not present in the context.
2. Assess your own confidence in this response as "high", "medium", or "low", based on whether the context actually answers the customer's specific question.

Respond ONLY in this exact JSON format, no other text:
{{
  "response": "<your drafted response>",
  "confidence": "<high|medium|low>",
  "confidence_reason": "<one short sentence>"
}}"""

    llm_response = llm.invoke(prompt)
    raw = llm_response.content.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        draft = parsed.get("response", "").strip()
        llm_confidence = parsed.get("confidence", "low").strip().lower()
        confidence_reason = parsed.get("confidence_reason", "")
    except (json.JSONDecodeError, AttributeError):
        # If the model didn't return valid JSON, fail safe: escalate
        draft = raw
        llm_confidence = "low"
        confidence_reason = "Could not parse model output as structured JSON."

    # --- Hybrid escalation logic ---
    score_flag = best_score is None or best_score > SCORE_ESCALATION_THRESHOLD
    llm_flag = llm_confidence == "low"

    escalate = score_flag or llm_flag

    reasons = []
    if score_flag:
        reasons.append(f"weak retrieval match (score={best_score})")
    if llm_flag:
        reasons.append(f"LLM low confidence: {confidence_reason}")

    state["draft_response"] = draft
    state["escalate"] = escalate
    state["escalation_reason"] = "; ".join(reasons) if escalate else None

    return state


if __name__ == "__main__":
    # Test 1: should be answerable (grounded in FAQ)
    test_state: AgentState = {
        "query": "I was charged twice for my subscription this month",
        "intent": "billing",
        "retrieved_docs": [
            {"question": "Why was I charged twice this month?",
             "answer": "Duplicate charges are usually due to a failed payment retry. Please check your billing history in Account Settings, or contact support with your transaction ID for a refund.",
             "score": 0.4534},
        ],
        "best_score": 0.4534,
        "draft_response": None,
        "escalate": None,
        "escalation_reason": None,
    }
    result = draft_response(test_state)
    print("=== TEST 1: Answerable query ===")
    print("Response:", result["draft_response"])
    print("Escalate:", result["escalate"])
    print("Reason:", result["escalation_reason"])

    # Test 2: should escalate (irrelevant/out-of-scope query)
    test_state_2: AgentState = {
        "query": "Can you help me plan my wedding menu?",
        "intent": "general_inquiry",
        "retrieved_docs": [
            {"question": "How do I reset my password?",
             "answer": "Go to the login page and click 'Forgot Password'.",
             "score": 1.6},
        ],
        "best_score": 1.6,
        "draft_response": None,
        "escalate": None,
        "escalation_reason": None,
    }
    result2 = draft_response(test_state_2)
    print("\n=== TEST 2: Out-of-scope query ===")
    print("Response:", result2["draft_response"])
    print("Escalate:", result2["escalate"])
    print("Reason:", result2["escalation_reason"])