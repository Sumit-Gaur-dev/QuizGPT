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
    col1, col2, col3, col4 = st.columns(4)

    num_pdfs = len(st.session_state.get("uploaded_files", []))
    num_quizzes = st.session_state.get("generated_quizzes_count", 0)
    completed_quizzes = len(st.session_state.get("quiz_history", []))

    extracted_docs = st.session_state.get("extracted_docs", {})
    successful_docs = [d for d in extracted_docs.values() if d.get("success")]
    total_pages = sum(d.get("extracted_pages_count", 0) for d in successful_docs)
    total_chunks = len(st.session_state.get("chunks", []))

    with col1:
        delta_label = f"{total_pages} pages extracted" if total_pages > 0 else ("Ready for upload" if num_pdfs == 0 else "Pending extraction")
        st.metric(label="📄 Uploaded PDFs", value=num_pdfs, delta=delta_label)

    with col2:
        st.metric(label="🧩 LangChain Chunks", value=total_chunks, delta="Ready for ChromaDB" if total_chunks > 0 else "None")

    with col3:
        st.metric(label="⚙️ Generated Quizzes", value=num_quizzes)

    with col4:
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
    st.subheader("🔍 The 6-Step RAG Journey")
    step1, step2, step3 = st.columns(3)

    with step1:
        st.info("**Step 1: Ingestion**\n\nPDF text is parsed page-by-page using PyMuPDF (`fitz`).")
        st.info("**Step 2: LangChain Chunking**\n\nText is modeled into Documents and split using `RecursiveCharacterTextSplitter`.")

    with step2:
        st.info("**Step 3: Embeddings**\n\nGemini `text-embedding-004` turns chunks into vectors.")
        st.info("**Step 4: Vector Storage**\n\nVectors and chunk metadata are saved in session ChromaDB.")

    with step3:
        st.info("**Step 5: Grounded Retrieval**\n\nTop-K relevant chunks are matched to your study query.")
        st.info("**Step 6: Evaluation**\n\nObjective questions scored in Python; short answers scored by AI.")

