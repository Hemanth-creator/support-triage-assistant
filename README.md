# Customer Support Triage & Response Assistant

A multi-agent LLM system that automates customer support triage — classifying incoming queries, retrieving grounded context from a knowledge base, and drafting responses, with automatic escalation when the system isn't confident enough to answer.

Built with **LangGraph**, **FastAPI**, **Groq (Llama-class models)**, and **FAISS**.

## Overview

Instead of a single monolithic LLM call, this project routes each customer query through a graph of specialized agents that hand off state to one another:

```
Customer Query
      │
      ▼
[Intent Classifier]  → classifies query into a support category
      │
      ▼
[Knowledge Lookup]   → retrieves relevant docs via FAISS vector search (RAG)
      │
      ▼
[Response Drafter]   → drafts a grounded answer, self-assesses confidence,
      │                 and flags escalation when needed
      ▼
  API Response
```

Each agent reads from and writes to a shared `AgentState` object as it flows through a LangGraph `StateGraph` — this keeps the pipeline modular, inspectable, and easy to extend (e.g. adding a sentiment-analysis agent or a branching router later).

## Key Features

- **Multi-agent architecture (LangGraph):** three specialized agents — intent classification, knowledge retrieval, and response drafting — communicate through a shared state graph rather than one large prompt.
- **Retrieval-Augmented Generation (RAG):** responses are grounded in a real FAQ knowledge base using a FAISS vector store, reducing hallucination compared to a raw LLM call.
- **Hybrid escalation logic:** a query is flagged for human escalation when *either* the FAISS similarity score is weak *or* the LLM explicitly reports low confidence in its own answer — combining a retrieval-based signal with model self-assessment.
- **Fast inference via Groq:** uses Groq's API with Llama-class / open-weight models (`openai/gpt-oss-20b` for classification, `openai/gpt-oss-120b` for drafting) for low-latency responses.
- **Production-style API:** deployed as a FastAPI service with request validation, structured JSON responses, and a health-check endpoint.

## Tech Stack

| Component            | Technology                          |
|-----------------------|--------------------------------------|
| Agent orchestration   | LangGraph                           |
| LLM inference         | Groq API (Llama-class / GPT-OSS models) |
| Vector search / RAG   | FAISS + Sentence-Transformers embeddings |
| API framework         | FastAPI + Uvicorn                   |
| Language              | Python                              |
| Deployment            | Render                              |

## Project Structure

```
support-triage-assistant/
├── app/
│   ├── agents/
│   │   ├── state.py              # Shared AgentState schema
│   │   ├── intent_classifier.py  # Agent 1: classifies query intent
│   │   ├── knowledge_lookup.py   # Agent 2: retrieves FAQ context via FAISS
│   │   ├── response_drafter.py   # Agent 3: drafts grounded response + escalation logic
│   │   └── graph.py              # LangGraph StateGraph wiring the agents together
│   ├── rag/
│   │   ├── build_index.py        # Builds the FAISS index from the FAQ dataset
│   │   └── retriever.py          # Loads the index and exposes retrieve()
│   ├── data/
│   │   ├── faq.json              # Sample support knowledge base
│   │   └── faiss_index/          # Generated vector store (built, not hand-written)
│   └── main.py                   # FastAPI app and endpoints
├── requirements.txt
└── README.md
```

## How It Works

1. **Intent Classification** — the query is sent to a Groq LLM with a fixed set of categories (`billing`, `account`, `technical`, `refund`, `general_inquiry`) and classified with `temperature=0` for consistency.
2. **Knowledge Lookup** — the query is embedded (via `sentence-transformers/all-MiniLM-L6-v2`) and matched against a FAISS index built from the FAQ knowledge base, returning the top-k most relevant entries and a similarity score.
3. **Response Drafting** — the retrieved context is passed to a Groq LLM, which is instructed to answer **only** using that context (to prevent hallucination) and to self-report its confidence (`high` / `medium` / `low`).
4. **Escalation Decision** — the query is flagged for human escalation if *either*:
   - the FAISS similarity score is above a set threshold (weak retrieval match), **or**
   - the LLM reports low confidence in its own answer.

## API

### `GET /health`
Simple health check endpoint.

### `POST /query`
Submit a customer query and receive a routed, drafted response.

**Request:**
```json
{
  "query": "I want to cancel my subscription"
}
```

**Response:**
```json
{
  "query": "I want to cancel my subscription",
  "intent": "account",
  "draft_response": "To cancel your subscription, go to Account Settings > Billing > Cancel Subscription. Your access will continue until the end of the current billing cycle.",
  "escalate": false,
  "escalation_reason": null,
  "best_score": 0.3023790121078491
}
```

Interactive API docs are available at `/docs` (Swagger UI) once the server is running.

## Running Locally

```bash
# Clone and enter the repo
git clone https://github.com/<your-username>/support-triage-assistant.git
cd support-triage-assistant

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Build the FAISS index from the FAQ knowledge base
python app/rag/build_index.py

# Run the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/docs` to try it out.


## Possible Extensions

- Branching graph logic based on intent (e.g. route billing queries through a different sub-chain than technical ones)
- Persist escalated queries to a database or ticketing system integration
- Add a sentiment-analysis agent to prioritize frustrated customers
- Swap the sample FAQ for a larger, real-world knowledge base
- Add conversation memory for multi-turn support threads