# Policy / SOP Assistant

A professional Streamlit + LangChain RAG application for answering questions over internal policies, SOPs, manuals, and handbooks.

## Features
- Upload multiple PDF policy/SOP documents
- Build a local Chroma vector database
- Ask grounded questions in a clean chat UI
- View source previews and page references
- Clear and rebuild the knowledge base

## Tech stack
- Streamlit
- LangChain
- OpenAI
- Chroma
- PyPDF

## Setup

### 1. Create environment
```bash
conda create -n policy-rag python=3.11 -y
conda activate policy-rag
python -m pip install --upgrade pip