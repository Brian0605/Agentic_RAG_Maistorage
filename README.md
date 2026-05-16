# Agentic RAG System

A fully local, intelligent document Q&A system built with LangGraph.

## Features
- Agentic loop with query rewriting and hallucination checking
- Citations with source filename and page number
- MMR optimized retrieval for diverse, accurate results
- 13/13 unit tests passing
- 100% local — no API key needed

## Tech Stack
- LangGraph — agent workflow
- LangChain — AI framework
- Ollama (llama3.2) — local LLM
- HuggingFace (all-MiniLM-L6-v2) — local embeddings
- ChromaDB — vector database
- Streamlit — web UI

## How to Run

### 1. Install Ollama
Download from https://ollama.com and pull the model:
ollama pull llama3.2

### 2. Install dependencies
pip install streamlit langchain langchain-community langchain-ollama
pip install langgraph chromadb pypdf sentence-transformers langchain-huggingface

### 3. Run the app
streamlit run app.py

### 4. Run tests
pytest test_agentic_rag.py -v

## System Architecture
Retrieve → Grade Docs → Generate → Hallucination Check → End
              ↓ (no docs)              ↓ (hallucinated)
           Rewrite Query          Regenerate (max 2 retries)