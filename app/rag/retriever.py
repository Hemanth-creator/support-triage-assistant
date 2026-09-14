"""
Loads the saved FAISS index and exposes a simple retrieval function
that other parts of the app (e.g. the knowledge_lookup agent) can call.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

_vectorstore = None
_embeddings = None


def _get_vectorstore():
    """Lazy-load the FAISS index and embedding model once, then reuse them."""
    global _vectorstore, _embeddings
    if _vectorstore is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectorstore = FAISS.load_local(
            INDEX_PATH,
            _embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(query: str, k: int = 3):
    """
    Given a customer query, return the top-k most relevant FAQ entries.

    Returns a list of dicts: [{"question": ..., "answer": ..., "score": ...}, ...]
    """
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=k)

    output = []
    for doc, score in results:
        output.append(
            {
                "question": doc.metadata.get("question"),
                "answer": doc.metadata.get("answer"),
                "score": float(score),
            }
        )
    return output


if __name__ == "__main__":
    # Quick manual test
    test_query = "I want my money back"
    results = retrieve(test_query, k=3)
    print(f"Query: {test_query}\n")
    for r in results:
        print(f"Score: {r['score']:.4f} | Q: {r['question']}")
        print(f"  A: {r['answer']}\n")