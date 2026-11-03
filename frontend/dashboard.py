"""
Dashboard view for Quiz RAG.
Displays key session metrics, pipeline overview, and quick links.
"""

import streamlit as st


def render_dashboard() -> None:
    """Render the main dashboard overview page."""
    st.header("📊 Application Dashboard")
    st.markdown(
        "Welcome to **Quiz RAG**! This application helps you transform lecture notes, textbooks, "
        "and PDFs into personalized study quizzes using Retrieval-Augmented Generation."
    )

    st.divider()

    # Metric summary row
    col1, col2, col3 = st.columns(3)

    num_pdfs = len(st.session_state.get("uploaded_files", []))
    num_quizzes = st.session_state.get("generated_quizzes_count", 0)
    completed_quizzes = len(st.session_state.get("quiz_history", []))

    with col1:
        st.metric(label="📄 Uploaded PDFs", value=num_pdfs, delta="Ready for indexing" if num_pdfs > 0 else "None")

    with col2:
        st.metric(label="⚙️ Generated Quizzes", value=num_quizzes)

    with col3:
        st.metric(label="🏆 Completed in History", value=completed_quizzes)

    st.divider()

    # Quick action banner
    st.subheader("🚀 Quick Actions")
    action_col1, action_col2 = st.columns(2)

    with action_col1:
        with st.container(border=True):
            st.markdown("### 📁 1. Upload Study Notes")
            st.write("Upload PDF notes from lectures, textbooks, or course handouts.")
            if st.button("Go to Uploads ➔", use_container_width=True):
                st.session_state.current_page = "📁 Upload Notes"
                st.rerun()

    with action_col2:
        with st.container(border=True):
            st.markdown("### ⚙️ 2. Generate a Quiz")
            st.write("Configure quiz difficulty, question formats, and target topics.")
            if st.button("Go to Generator ➔", use_container_width=True):
                st.session_state.current_page = "⚙️ Generate Quiz"
                st.rerun()

    st.markdown("---")

    # Visible RAG Pipeline Walkthrough
    st.subheader("🔍 The 6-Step RAG Journey (Phase 1 UI View)")
    step1, step2, step3 = st.columns(3)

    with step1:
        st.info("**Step 1: Ingestion**\n\nPDF text is parsed page-by-page using PyMuPDF (`fitz`).")
        st.info("**Step 2: Chunking**\n\nText is split into ~1,000 char segments with 150 char overlap.")

    with step2:
        st.info("**Step 3: Embeddings**\n\nGemini `text-embedding-004` turns chunks into vectors.")
        st.info("**Step 4: Vector Storage**\n\nVectors and chunk metadata are saved in session ChromaDB.")

    with step3:
        st.info("**Step 5: Grounded Retrieval**\n\nTop-K relevant chunks are matched to your study query.")
        st.info("**Step 6: Evaluation**\n\nObjective questions scored in Python; short answers scored by AI.")

