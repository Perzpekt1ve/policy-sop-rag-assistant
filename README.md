# Policy / SOP Assistant

A professional **Retrieval-Augmented Generation (RAG)** application that allows users to upload internal **policy documents, SOPs, manuals, and handbooks**, ask natural language questions, and receive grounded answers with source references.

This project is built with **Streamlit**, **LangChain**, **Chroma**, **Ollama**, and **Hugging Face embeddings**, and is designed to run locally on lightweight hardware.

---

## Overview

Organizations often store important information inside long PDF documents such as:

- HR policies
- employee handbooks
- standard operating procedures (SOPs)
- safety manuals
- compliance documents
- procurement and approval workflows

Manually searching these documents is slow and inefficient.

This project solves that problem by building an AI assistant that:

1. ingests PDF documents,
2. breaks them into searchable chunks,
3. stores them in a vector database,
4. retrieves relevant sections based on a user query,
5. generates an answer grounded in the retrieved context.

---

## Features

- Upload multiple PDF policy/SOP documents
- Build a local semantic search index
- Ask questions in natural language
- Retrieve relevant policy sections using vector search
- Generate answers using a local LLM through Ollama
- Display source references used to generate the answer
- Clean Streamlit chat interface
- Fully local and free to run after setup

---

## Tech Stack

### Frontend
- **Streamlit**

### RAG / Orchestration
- **LangChain**

### Embeddings
- **Hugging Face Sentence Transformers**
- Model: `sentence-transformers/all-MiniLM-L6-v2`

### Vector Database
- **Chroma**

### LLM
- **Ollama**
- Suggested chat model: `qwen2.5:1.5b`

### Document Loading
- **PyPDF**

---

## Architecture

### High-Level Flow

```text
User uploads PDFs
        ↓
PDF text extraction
        ↓
Text chunking
        ↓
Embedding generation
        ↓
Chroma vector database
        ↓
User asks question
        ↓
Relevant chunks retrieved
        ↓
Local LLM generates grounded answer
        ↓
Answer + source references shown in UI
