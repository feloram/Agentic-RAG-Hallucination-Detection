# Project Details

## Project Title

Design and Evaluation of an Agentic Retrieval-Augmented Generation (RAG) System with Automatic Hallucination Detection for Reliable Question Answering

## Student

Fatemeh Sadat Mirshaki

## Supervisor

Dr. Jalali

## University

University of Qom

---

## 1. Project Overview

This project presents an Agentic Retrieval-Augmented Generation (RAG) system designed for reliable scientific question answering.

The system retrieves relevant evidence from a collection of scientific documents and evaluates the retrieved evidence before generating an answer.

The main objective is to reduce unsupported answers and improve the reliability of responses generated from scientific documents.

---

## 2. Main Problem

Large Language Models may generate hallucinated information.

A hallucination occurs when a model produces information that is unsupported, inaccurate, or not grounded in the available evidence.

This problem is especially important in scientific question answering, where answers should be supported by reliable documents.

---

## 3. Proposed Approach

The system follows an Agentic RAG architecture.

The main workflow includes:

1. Query Analysis
2. Dense Retrieval
3. Re-ranking
4. Evidence Filtering
5. Evidence Quality Evaluation
6. Context Construction
7. Answer Generation
8. Claim-Level Hallucination Detection

---

## 4. Retrieval Component

The system uses dense vector retrieval to identify semantically relevant document chunks.

Scientific documents are divided into smaller text chunks.

Each chunk is converted into a dense vector representation.

The user query is also converted into a vector.

Similarity is calculated between the query vector and document embeddings.

The most relevant candidates are selected for further processing.

---

## 5. Embedding Model

The project uses:

BAAI/bge-base-en-v1.5

This model is used to generate dense semantic embeddings for queries and document chunks.

---

## 6. Re-ranking Component

After initial retrieval, candidate documents are re-ranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

The cross-encoder directly evaluates the relevance between the user query and each retrieved text.

This improves the ordering of retrieved evidence.

---

## 7. Evidence Quality Evaluation

The system evaluates evidence using:

- Number of retrieved evidence blocks
- Dense similarity score
- Relevance of retrieved content

Evidence quality is categorized as:

- Sufficient
- Moderate
- Insufficient

The system can perform multiple retrieval attempts when evidence quality is not sufficient.

---

## 8. Answer Generation

The current answer generation module follows an extractive approach.

The system:

1. Splits the retrieved context into sentences.
2. Extracts important terms from the query.
3. Measures the overlap between query terms and sentence terms.
4. Ranks sentences according to relevance.
5. Selects the most relevant sentences as the final answer.

This approach helps keep the answer grounded in retrieved evidence.

---

## 9. Hallucination Detection

The project evaluates generated answers at the claim level.

The answer is divided into individual claims.

Each claim is compared with the retrieved evidence.

Claims are categorized according to their level of support.

The evaluation includes:

- Supported Claims
- Uncertain Claims
- Possible Hallucinations

---

## 10. User Interface

The system provides a Streamlit-based interface.

The interface displays:

- User query
- Workflow status
- Retrieval attempts
- Evidence quality
- Evidence score
- Final answer
- Retrieved evidence
- Similarity scores
- Re-ranker scores

---

## 11. Technologies

The main technologies used in this project include:

- Python
- Streamlit
- NumPy
- Sentence Transformers
- Cross-Encoder
- PyMuPDF
- Dense Vector Embeddings
- Retrieval-Augmented Generation
- Agentic Workflow

---

## 12. Project Architecture

The general architecture is:

User Query
↓
Query Analysis
↓
Dense Retrieval
↓
Re-ranking
↓
Evidence Filtering
↓
Evidence Quality Evaluation
↓
Context Construction
↓
Answer Generation
↓
Claim-Level Hallucination Detection
↓
Reliable Evidence-Based Answer

---

## 13. Strengths

The main strengths of the system include:

- Evidence-based answer generation
- Modular Agentic RAG architecture
- Dense semantic retrieval
- Cross-encoder re-ranking
- Evidence quality assessment
- Claim-level hallucination evaluation
- Interactive Streamlit interface
- Support for scientific documents

---

## 14. Limitations

The current system has several limitations:

- The answer generation module is extractive.
- The quality of the final answer depends on the retrieved evidence.
- Retrieval quality can be affected by document chunking.
- The system does not currently use a large generative language model for final answer synthesis.
- Claim-level evaluation may classify some claims as uncertain when evidence is incomplete.

---

## 15. Future Improvements

Possible future improvements include:

- Integrating a generative LLM for grounded answer synthesis.
- Improving query refinement.
- Adding hybrid retrieval.
- Improving claim-evidence alignment.
- Adding more advanced hallucination detection models.
- Adding automated evaluation metrics.
- Supporting larger document collections.
