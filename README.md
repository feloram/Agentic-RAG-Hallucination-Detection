# Agentic-RAG-Hallucination-Detection
# Design and Evaluation of an Agentic RAG System with Automatic Hallucination Detection

## Project Overview

This project presents an Agentic Retrieval-Augmented Generation (RAG) system designed for reliable scientific question answering.

The system retrieves relevant evidence from scientific documents, re-ranks the retrieved results, filters redundant evidence, evaluates evidence quality, and generates an evidence-based answer.

The project also includes an automatic claim-level hallucination detection and evaluation pipeline.

---

## Main Objectives

- Retrieve relevant scientific evidence from PDF documents
- Improve retrieval quality using dense embeddings
- Re-rank retrieved evidence using a Cross-Encoder
- Filter duplicate and low-quality evidence
- Evaluate evidence quality automatically
- Generate extractive, evidence-based answers
- Detect unsupported or uncertain claims
- Evaluate hallucination-related behavior automatically

---

## System Architecture

The system follows an Agentic RAG workflow:

PDF Documents
        ↓
Text Extraction and Chunking
        ↓
Dense Embedding Generation
        ↓
Vector Retrieval
        ↓
Cross-Encoder Re-ranking
        ↓
Evidence Filtering
        ↓
Evidence Quality Evaluation
        ↓
Context Construction
        ↓
Answer Generation
        ↓
Claim-Level Evaluation
        ↓
Hallucination Detection

---

## Main Technologies

- Python
- Streamlit
- NumPy
- PyTorch
- Sentence Transformers
- BGE Embeddings
- Cross-Encoder Re-ranking
- PyMuPDF
- Scikit-learn

---

## Models

### Embedding Model

BAAI/bge-base-en-v1.5

Used for generating dense vector representations of scientific text and user queries.

### Re-ranking Model

cross-encoder/ms-marco-MiniLM-L-6-v2

Used to re-rank retrieved candidate documents according to their relevance to the query.

---

## Agentic RAG Workflow

The system performs the following steps:

1. Query analysis
2. Evidence retrieval
3. Re-ranking
4. Evidence filtering
5. Evidence quality evaluation
6. Context construction
7. Extractive answer generation
8. Claim-level evaluation
9. Hallucination detection

---

## Evaluation

The system evaluates generated answers using claim-level analysis.

Each claim can be evaluated according to its relationship with the retrieved evidence.

The evaluation includes:

- Supported Claims
- Uncertain Claims
- Hallucination-related Analysis
- Support Score
- Hallucination Score

---

## User Interface

A Streamlit interface is provided for interacting with the system.

Users can:

- Enter a scientific question
- Run the Agentic RAG workflow
- View the generated answer
- Inspect retrieved evidence
- View evidence quality
- View retrieval statistics
- Examine the system evaluation results

---

## Project Structure

```text
ProjectNLP/
│
├── app.py
├── rag_backend.py
├── requirements.txt
├── .gitignore
└── README.md
