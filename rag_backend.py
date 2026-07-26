
import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    "/content/drive/MyDrive/ProjectNLP"
)

PROCESSED_DIR = PROJECT_DIR / "processed"

EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"

CHUNKS_JSONL_PATH = (
    PROCESSED_DIR / "chunks.jsonl"
)

EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "chunk_embeddings.npy"
)

METADATA_PATH = (
    EMBEDDINGS_DIR / "embedding_metadata.jsonl"
)

# ============================================================
# LOAD DATA
# ============================================================

chunks = []

with open(
    CHUNKS_JSONL_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():

            chunks.append(
                json.loads(line)
            )


embeddings = np.load(
    EMBEDDINGS_PATH
)


metadata = []

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():

            metadata.append(
                json.loads(line)
            )

# ============================================================
# LOAD MODELS
# ============================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_top_k(
    query,
    top_k=10
):

    query_vector = (
        embedding_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
    )

    similarities = (
        embeddings @ query_vector
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({

            "chunk_id":
                metadata[index]["chunk_id"],

            "file_name":
                metadata[index]["file_name"],

            "page_start":
                metadata[index]["page_start"],

            "page_end":
                metadata[index]["page_end"],

            "text":
                chunks[index]["text"],

            "similarity_score":
                float(
                    similarities[index]
                )

        })

    return results


# ============================================================
# RE-RANKING
# ============================================================

def rerank_results(
    query,
    candidates
):

    if not candidates:

        return []

    pairs = [

        (
            query,
            result["text"]
        )

        for result in candidates

    ]

    scores = reranker.predict(
        pairs
    )

    for index, result in enumerate(
        candidates
    ):

        result["reranker_score"] = (
            float(scores[index])
        )

    return sorted(

        candidates,

        key=lambda x:
            x["reranker_score"],

        reverse=True

    )


# ============================================================
# AGENT STATE
# ============================================================

@dataclass
class AgentState:

    query: str

    original_query: str = ""

    retrieval_attempts: int = 0

    max_retrieval_attempts: int = 3

    retrieved_candidates: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    reranked_results: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    filtered_evidence: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    evidence_quality: str = (
        "unknown"
    )

    evidence_score: float = 0.0

    final_context: str = ""

    generated_answer: str = ""

    status: str = "initialized"


# ============================================================
# EVIDENCE FILTERING
# ============================================================

def filter_evidence(
    results,
    max_results=5
):

    filtered = []

    seen = set()

    for result in results:

        text = result.get(
            "text",
            ""
        ).strip()

        if not text:

            continue

        signature = (
            text[:300]
            .lower()
            .strip()
        )

        if signature in seen:

            continue

        seen.add(signature)

        filtered.append(
            result
        )

        if len(filtered) >= max_results:

            break

    return filtered


# ============================================================
# EVIDENCE QUALITY
# ============================================================

def evaluate_evidence(
    evidence
):

    if not evidence:

        return (
            "insufficient",
            0.0
        )

    scores = [

        float(
            item.get(
                "similarity_score",
                0.0
            )
        )

        for item in evidence

    ]

    average_score = float(
        np.mean(scores)
    )

    if (

        len(evidence) >= 3
        and
        average_score >= 0.45

    ):

        quality = "sufficient"

    elif (

        len(evidence) >= 2
        and
        average_score >= 0.30

    ):

        quality = "moderate"

    else:

        quality = "insufficient"

    return (

        quality,
        round(
            average_score,
            4
        )

    )


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    evidence
):

    blocks = []

    for index, item in enumerate(
        evidence,
        start=1
    ):

        blocks.append(

            f"[Evidence {index}]\n"

            f"Document: "
            f"{item.get('file_name', 'Unknown')}\n"

            f"Pages: "
            f"{item.get('page_start', '?')}-"
            f"{item.get('page_end', '?')}\n\n"

            f"{item.get('text', '')}"

        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# ANSWER GENERATION — FINAL IMPROVED EXTRACTIVE VERSION
# ============================================================

def generate_answer(
    query,
    context,
    max_sentences=5
):

    if not context:

        return (
            "No sufficient evidence was found "
            "to answer the question."
        )

    # ========================================================
    # 1. REMOVE EVIDENCE METADATA
    # ========================================================

    clean_context = re.sub(
        r"\[Evidence\s+\d+\]",
        "",
        context,
        flags=re.IGNORECASE
    )

    clean_context = re.sub(
        r"Document:\s*.*?\n",
        "",
        clean_context,
        flags=re.IGNORECASE
    )

    clean_context = re.sub(
        r"Pages:\s*.*?\n",
        "",
        clean_context,
        flags=re.IGNORECASE
    )

    # ========================================================
    # 2. SENTENCE SEGMENTATION
    # ========================================================

    sentences = re.split(
        r"(?<=[.!?])\s+",
        clean_context
    )

    # ========================================================
    # 3. QUERY ANALYSIS
    # ========================================================

    query_lower = query.lower()

    method_question = any(

        phrase in query_lower

        for phrase in [

            "what methods",
            "which methods",
            "what techniques",
            "which techniques",
            "what approaches",
            "which approaches",
            "how is",
            "how are"

        ]

    )

    query_terms = set(

        re.findall(

            r"\b[a-zA-Z]{3,}\b",

            query_lower

        )

    )

    stopwords = {

        "what",
        "which",
        "how",
        "why",
        "when",
        "where",
        "are",
        "is",
        "the",
        "for",
        "and",
        "with",
        "used",
        "use",
        "this",
        "that",
        "from",
        "into",
        "their",
        "these",
        "those"

    }

    query_terms -= stopwords

    # ========================================================
    # 4. DOMAIN TERMS
    # ========================================================

    method_terms = {

        "method",
        "methods",
        "approach",
        "approaches",
        "technique",
        "techniques",
        "framework",
        "model",
        "models",
        "consistency",
        "uncertainty",
        "probabilistic",
        "probability",
        "distance",
        "distances",
        "mmd",
        "entropy",
        "eigenscore",
        "eigen",
        "hidden",
        "state",
        "states",
        "token",
        "confidence",
        "black-box",
        "gray-box",
        "white-box",
        "blackbox",
        "graybox",
        "whitebox",
        "retrieval",
        "rag",
        "factuality"

    }

    # ========================================================
    # 5. IRRELEVANT / BIBLIOGRAPHIC PATTERNS
    # ========================================================

    author_patterns = [

        "e-mail:",
        "email:",
        "department of",
        "university",
        "laboratory",
        "lab,",
        "institute",
        "markov lab",
        "ai center",
        "sber,",
        "skoltech",
        "saint-petersburg"

    ]

    metadata_patterns = [

        "http://",
        "https://",
        "arxiv:",
        "doi:",
        "isbn",
        "copyright",
        "all rights reserved"

    ]

    # ========================================================
    # 6. SCORE SENTENCES
    # ========================================================

    scored_sentences = []

    for index, sentence in enumerate(sentences):

        sentence = sentence.strip()

        if len(sentence) < 45:

            continue

        sentence_lower = sentence.lower()

        sentence_terms = set(

            re.findall(

                r"\b[a-zA-Z]{3,}\b",

                sentence_lower

            )

        )

        # ----------------------------------------------------
        # Query overlap
        # ----------------------------------------------------

        query_overlap = len(

            query_terms.intersection(

                sentence_terms

            )

        )

        # ----------------------------------------------------
        # Method-related overlap
        # ----------------------------------------------------

        method_overlap = len(

            method_terms.intersection(

                sentence_terms

            )

        )

        # ----------------------------------------------------
        # Methodology signals
        # ----------------------------------------------------

        methodology_score = 0

        methodology_patterns = [

            "methods include",
            "methods can be",
            "methods are",
            "approaches include",
            "approaches can be",
            "techniques include",
            "can be divided into",
            "can be grouped into",
            "broadly divided",
            "broadly grouped",
            "categories",
            "based methods",
            "based approaches",
            "rely on",
            "measure",
            "estimate",
            "detect",
            "detection",
            "used for hallucination detection",
            "hallucination detection methods"

        ]

        for pattern in methodology_patterns:

            if pattern in sentence_lower:

                methodology_score += 3

        # ----------------------------------------------------
        # Method question bonus
        # ----------------------------------------------------

        method_question_bonus = 0

        if method_question:

            if method_overlap >= 2:

                method_question_bonus += 5

            if (

                "method" in sentence_lower

                or

                "approach" in sentence_lower

                or

                "technique" in sentence_lower

                or

                "category" in sentence_lower

                or

                "grouped" in sentence_lower

                or

                "divided" in sentence_lower

            ):

                method_question_bonus += 4

        # ----------------------------------------------------
        # Penalties
        # ----------------------------------------------------

        penalty = 0

        for pattern in author_patterns:

            if pattern in sentence_lower:

                penalty += 15

        for pattern in metadata_patterns:

            if pattern in sentence_lower:

                penalty += 15

        # Penalize isolated result statements
        if sentence_lower.startswith(

            (

                "consequently",

                "as for",

                "the results",

                "these results",

                "performance in"

            )

        ):

            penalty += 5

        # Penalize very long bibliographic sentences
        if (

            len(sentence) > 500

            and

            (

                "e-mail" in sentence_lower

                or

                "department" in sentence_lower

                or

                "university" in sentence_lower

            )

        ):

            penalty += 20

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (

            query_overlap * 3

            +

            method_overlap * 2

            +

            methodology_score

            +

            method_question_bonus

            -

            penalty

        )

        scored_sentences.append(

            {

                "score": final_score,

                "query_overlap": query_overlap,

                "method_overlap": method_overlap,

                "index": index,

                "sentence": sentence

            }

        )

    # ========================================================
    # 7. SORT BY RELEVANCE
    # ========================================================

    scored_sentences.sort(

        key=lambda item: (

            item["score"],

            item["method_overlap"],

            item["query_overlap"]

        ),

        reverse=True

    )

    # ========================================================
    # 8. SELECT DIVERSE, RELEVANT SENTENCES
    # ========================================================

    selected = []

    selected_signatures = set()

    for item in scored_sentences:

        sentence = item["sentence"]

        signature = (

            sentence[:150]

            .lower()

            .strip()

        )

        if signature in selected_signatures:

            continue

        selected_signatures.add(signature)

        # Avoid extremely low-quality sentences
        if item["score"] <= 0:

            continue

        selected.append(sentence)

        if len(selected) >= max_sentences:

            break

    # ========================================================
    # 9. FALLBACK
    # ========================================================

    if not selected:

        return (

            "The retrieved evidence does not provide "

            "a sufficiently clear answer."

        )

    # ========================================================
    # 10. ORDER ANSWER LOGICALLY
    # ========================================================

    if method_question:

        selected.sort(

            key=lambda sentence: (

                "methods can be" not in sentence.lower(),

                "methods include" not in sentence.lower(),

                "approaches include" not in sentence.lower(),

                "can be divided into" not in sentence.lower(),

                "can be grouped into" not in sentence.lower()

            )

        )

    # ========================================================
    # 11. FINAL EXTRACTIVE ANSWER
    # ========================================================

    return " ".join(selected)
# ============================================================
# MAIN WORKFLOW
# ============================================================

def run_agentic_rag_workflow(
    query,
    max_attempts=3
):

    state = AgentState(

        query=query,

        original_query=query,

        max_retrieval_attempts=
            max_attempts

    )

    while (

        state.retrieval_attempts
        <
        state.max_retrieval_attempts

    ):

        state.retrieval_attempts += 1

        candidates = retrieve_top_k(
            query=state.query,
            top_k=10
        )

        state.retrieved_candidates = (
            candidates
        )

        reranked = rerank_results(
            query=state.query,
            candidates=candidates
        )

        state.reranked_results = (
            reranked
        )

        filtered = filter_evidence(
            reranked
        )

        state.filtered_evidence = (
            filtered
        )

        quality, score = (
            evaluate_evidence(
                filtered
            )
        )

        state.evidence_quality = (
            quality
        )

        state.evidence_score = (
            score
        )

        if quality == "sufficient":

            break

    state.final_context = (
        build_context(
            state.filtered_evidence
        )
    )

    state.generated_answer = (
        generate_answer(
            query=state.query,
            context=state.final_context
        )
    )

    if state.final_context:

        state.status = (
            "completed_successfully"
        )

    else:

        state.status = (
            "completed_without_context"
        )

    return state
