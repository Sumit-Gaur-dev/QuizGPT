"""
Quiz generation view for Quiz RAG.
Configures quiz settings and generates strictly grounded questions using
LangChain and Gemini (or verified offline mock mode during development).
"""

import streamlit as st
from config.settings import has_valid_gemini_key, GEMINI_MODEL
from rag.generation import generate_quiz
from rag.vector_store import get_collection_stats


def render_generate_quiz() -> None:
    """Render the quiz generation configuration controls and handle generation."""
    st.header("⚙️ Generate a Study Quiz")
    st.markdown(
        "Generate a structured quiz strictly grounded in your uploaded study notes. "
        "Questions cite their exact source PDF and page number."
    )

    st.divider()

    # 1. Pipeline & API Status Indicator
    key_configured = has_valid_gemini_key()
    col_status, col_docs = st.columns(2)

    with col_status:
        with st.container(border=True):
            if key_configured:
                st.markdown(f"🤖 **Gemini Status:** `Connected ({GEMINI_MODEL})`")
                mock_default = st.session_state.get("force_mock_mode", False)
                force_mock = st.checkbox(
                    "🔒 Force Offline Mock Mode (Zero Gemini API calls)",
                    value=mock_default,
                    help="Enable to test quiz generation locally without consuming Gemini API quota."
                )
                st.session_state.force_mock_mode = force_mock
            else:
                st.markdown("🔒 **Gemini Status:** `Offline Mock Mode (Zero API calls)`")
                st.caption(
                    "No valid Gemini API key found in `.env`. Running safely in local mock mode. "
                    "Questions are strictly grounded in your note chunks with zero external API calls."
                )
                st.session_state.force_mock_mode = True

    with col_docs:
        with st.container(border=True):
            stats = get_collection_stats(st.session_state.session_id)
            total_chunks = stats["total_chunks"]
            st.markdown(f"📚 **Context Base:** `{total_chunks} ChromaDB chunks`")
            if total_chunks == 0:
                st.caption("⚠️ No documents indexed yet. A sample study notes context will be used for mock generation.")
            else:
                st.caption(f"Embedding Provider: `{stats['display_name']}`")

    st.write("")

    # 2. Configuration Form
    with st.form("quiz_generation_form"):
        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                label="Target Topic or Focus Area (Optional)",
                placeholder="e.g., Vector embeddings, RAG vs Fine-tuning, Chunking",
                help="Leave blank to generate a broad quiz across all uploaded notes."
            )

            difficulty = st.selectbox(
                label="Difficulty Level",
                options=["Easy", "Medium", "Hard"],
                index=1,
                help="Adjusts the depth and complexity of questions."
            )

        with col2:
            question_types = st.multiselect(
                label="Question Types",
                options=["MCQ", "True/False", "Short Answer"],
                default=["MCQ", "True/False", "Short Answer"],
                help="Choose which question formats should be included in the quiz."
            )

            num_questions = st.selectbox(
                label="Number of Questions",
                options=[5, 10, 15, 20],
                index=0,
                help="Select how many questions you want in this quiz."
            )

        submit_btn = st.form_submit_button(
            "🚀 Generate Quiz",
            type="primary",
            use_container_width=True
        )

    # 3. Handle Form Submission
    if submit_btn:
        if not question_types:
            st.error("Please select at least one question type (MCQ, True/False, or Short Answer).")
            return

        with st.spinner("Retrieving relevant context from ChromaDB and generating questions..."):
            try:
                quiz = generate_quiz(
                    session_id=st.session_state.session_id,
                    topic=topic,
                    difficulty=difficulty,
                    question_types=question_types,
                    num_questions=num_questions,
                    force_mock=st.session_state.get("force_mock_mode", True)
                )

                # Store active quiz in session state
                st.session_state.current_quiz = quiz
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.active_question_idx = 0
                st.session_state.latest_evaluation = None
                st.session_state.show_submit_confirm = False
                st.session_state.generated_quizzes_count += 1

                mode_label = "Offline Mock Mode (Zero API calls)" if st.session_state.get("force_mock_mode") else f"Gemini ({GEMINI_MODEL})"
                st.success(
                    f"🎉 Quiz **'{quiz['title']}'** generated successfully with {len(quiz['questions'])} questions!\n\n"
                    f"- **Difficulty:** {difficulty}\n"
                    f"- **Mode:** {mode_label}"
                )

            except Exception as e:
                st.error(f"❌ Quiz generation failed: {e}")
                return

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.info("Your quiz is ready! Click **Start Quiz Now** to begin the interactive assessment.")
        with col_b:
            if st.button("Start Quiz Now ➔", type="primary", use_container_width=True):
                st.session_state.current_page = "✍️ Attempt Quiz"
                st.rerun()

    st.markdown("---")

    # Informational notice
    with st.expander("ℹ️ Grounded Quiz Generation Architecture", expanded=False):
        st.markdown(
            """
            - **Strict Grounding:** Every question is retrieved from your study notes in ChromaDB with source filename and page numbers preserved.
            - **Zero Hallucination:** Prompts instruct the model to rely solely on retrieved context.
            - **Offline Mock Support:** If `GOOGLE_API_KEY` is not present, the system runs locally with zero external network requests while maintaining the exact same Pydantic data schemas.
            """
        )
