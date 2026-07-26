
import streamlit as st
import sys
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic RAG Hallucination Detection",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path("/content/drive/MyDrive/ProjectNLP")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# ============================================================
# IMPORT BACKEND
# ============================================================

try:
    from rag_backend import run_agentic_rag_workflow
except Exception as e:
    st.error("Backend could not be loaded.")
    st.exception(e)
    st.stop()

# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔍 Agentic RAG Hallucination Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'A Retrieval-Augmented Generation system for evidence-based scientific question answering'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write("📚 Scientific Documents: 15")
    st.write("📄 Total Pages: 262")
    st.write("🧩 Retrieval: Dense Embeddings")
    st.write("🔄 Re-ranking: Cross-Encoder")
    st.write("🤖 Architecture: Agentic RAG")

    st.divider()

    st.info(
        "The system retrieves scientific evidence, "
        "evaluates its quality, and generates an extractive answer."
    )

# ============================================================
# QUERY INPUT
# ============================================================

st.subheader("💬 Ask a Scientific Question")

query = st.text_area(
    "Enter your question:",
    placeholder=(
        "Example: How can small language models be used "
        "to detect hallucinations?"
    ),
    height=120
)

run_button = st.button(
    "🚀 Run Agentic RAG",
    use_container_width=True
)

# ============================================================
# WORKFLOW EXECUTION
# ============================================================

if run_button:

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner(
            "Running Agentic RAG workflow..."
        ):

            try:

                state = run_agentic_rag_workflow(
                    query=query.strip()
                )

                st.session_state["state"] = state

            except Exception as e:

                st.error(
                    "An error occurred during workflow execution."
                )

                st.exception(e)

# ============================================================
# DISPLAY RESULTS
# ============================================================

if "state" in st.session_state:

    state = st.session_state["state"]

    st.divider()

    st.header("📊 Workflow Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Status",
            state.status
        )

    with col2:

        st.metric(
            "Retrieval Attempts",
            state.retrieval_attempts
        )

    with col3:

        st.metric(
            "Evidence Quality",
            state.evidence_quality
        )

    with col4:

        st.metric(
            "Evidence Score",
            f"{state.evidence_score:.4f}"
        )

    st.divider()

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    st.header("📝 Final Answer")

    if state.generated_answer:

        st.success(
            state.generated_answer
        )

    else:

        st.warning(
            "No reliable answer was generated."
        )

     # ========================================================
    # EVIDENCE
    # ========================================================

    st.header("📚 Retrieved Evidence")

    if state.filtered_evidence:

        for index, evidence in enumerate(
            state.filtered_evidence,
            start=1
        ):

            file_name = evidence.get(
                "file_name",
                "Unknown Document"
            )

            page_start = evidence.get(
                "page_start",
                "?"
            )

            page_end = evidence.get(
                "page_end",
                "?"
            )

            similarity_score = float(
                evidence.get(
                    "similarity_score",
                    0.0
                )
            )

            reranker_score = float(
                evidence.get(
                    "reranker_score",
                    0.0
                )
            )

            # --------------------------------------------
            # Evidence Card
            # --------------------------------------------

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### 🔹 Evidence {index}"
                )

                st.markdown(
                    f"**📄 Document:** `{file_name}`"
                )

                col_a, col_b, col_c = st.columns(3)

                with col_a:

                    st.metric(
                        "📑 Pages",
                        f"{page_start}–{page_end}"
                    )

                with col_b:

                    st.metric(
                        "🔍 Dense Similarity",
                        f"{similarity_score:.4f}"
                    )

                with col_c:

                    st.metric(
                        "⚡ Re-ranker Score",
                        f"{reranker_score:.4f}"
                    )

                st.markdown(
                    "#### 📖 Evidence Text"
                )

                st.markdown(
                    evidence.get(
                        "text",
                        "No evidence text available."
                    )
                )

                st.divider()

    else:

        st.warning(
            "No evidence was retrieved."
        )
