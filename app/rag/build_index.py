"""
Builds a FAISS vector store from the FAQ knowledge base.
Run this once (or whenever faq.json changes) to (re)generate the index.
"""

import json
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faq.json")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")


def load_faq_documents():
    with open(DATA_PATH, "r") as f:
        faq_items = json.load(f)

    documents = []
    for item in faq_items:
        content = f"Q: {item['question']}\nA: {item['answer']}"
        documents.append(
            Document(
                page_content=content,
                metadata={"question": item["question"], "answer": item["answer"]},
            )
        )
    return documents


def build_and_save_index():
    print("Loading FAQ documents...")
    documents = load_faq_documents()
    print(documents)
    print(f"Loaded {len(documents)} FAQ entries.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to: {INDEX_PATH}")


if __name__ == "__main__":
    build_and_save_index()