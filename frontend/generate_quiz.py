"""
Quiz generation view for Quiz RAG.
Configures quiz settings and generates mock questions (Phase 1 mock).
"""

import streamlit as st
from config.session import get_mock_questions


def render_generate_quiz() -> None:
    """Render the quiz generation configuration controls."""
    st.header("⚙️ Generate a Study Quiz")
    st.markdown(
        "Customize your quiz parameters. In Phase 1, clicking **Generate Quiz** builds "
        "a mock quiz using your chosen settings and places it in the session state ready for an attempt."
    )

    st.divider()

    # Check if notes are present
    num_files = len(st.session_state.get("uploaded_files", []))
    if num_files == 0:
        st.warning("⚠️ No documents uploaded yet. You can still generate a sample quiz using mock notes!")
    else:
        st.success(f"📚 Grounding context: {num_files} document(s) available in your session.")

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
                help="Adjusts the complexity and depth of generated questions."
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

    if submit_btn:
        if not question_types:
            st.error("Please select at least one question type.")
            return

        # Generate mock questions matching user selections
        quiz_title = f"{topic} Quiz" if topic.strip() else "Study Notes Comprehensive Quiz"
        mock_qs = get_mock_questions(
            count=num_questions,
            difficulty=difficulty,
            question_types=question_types
        )

        st.session_state.current_quiz = {
            "title": quiz_title,
            "topic": topic if topic.strip() else "All Uploaded Notes",
            "difficulty": difficulty,
            "total_questions": len(mock_qs),
            "questions": mock_qs
        }

        # Reset active attempt states
        st.session_state.user_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.generated_quizzes_count += 1

        st.success(
            f"🎉 Quiz '{quiz_title}' generated successfully with {len(mock_qs)} questions! "
            f"Difficulty: **{difficulty}**"
        )

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.info("You can now navigate to **✍️ Attempt Quiz** or click the button on the right.")
        with col_b:
            if st.button("Start Quiz Now ➔", type="primary", use_container_width=True):
                st.session_state.current_page = "✍️ Attempt Quiz"
                st.rerun()

    st.markdown("---")

    # Informational notice
    with st.expander("ℹ️ Phase 1 Note: Mock Generation Details", expanded=False):
        st.markdown(
            """
            - **Current Phase (Phase 1):** Questions are created from an internal mock question bank matching your chosen formats and count.
            - **Upcoming Phase (Phase 5):** Gemini 2.5 Flash will be queried using the top-K retrieved vector chunks to produce structured, strictly grounded questions.
            """
        )

