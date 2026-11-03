"""
Quiz history view for Quiz RAG.
Displays previous quiz attempts, scores, and completion history.
"""

import streamlit as st


def render_quiz_history() -> None:
    """Render the quiz history and past attempts summary."""
    st.header("📜 Quiz History & Past Attempts")
    st.markdown(
        "View records of all your generated and completed quizzes from this session. "
        "Scores and question counts are recorded in temporary `st.session_state`."
    )

    st.divider()

    history = st.session_state.get("quiz_history", [])

    if not history:
        st.info("No quiz attempts recorded yet. Attempt a quiz to see it listed here.")
        if st.button("Go to Quiz Generator ➔", type="primary"):
            st.session_state.current_page = "⚙️ Generate Quiz"
            st.rerun()
        return

    # Metric summaries
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Quizzes Taken", value=len(history))
    with col2:
        st.metric(label="Latest Score", value=history[0]["score"])
    with col3:
        st.metric(label="Session Storage", value="Active (In-Memory)")

    st.divider()

    st.subheader("📋 Completed Quizzes")

    for idx, item in enumerate(history):
        with st.container(border=True):
            c_title, c_score, c_meta = st.columns([3, 1.5, 2.5])

            with c_title:
                st.markdown(f"🎯 **{item['title']}**")
                st.caption(f"Topic: `{item.get('topic', 'General')}` | Difficulty: `{item.get('difficulty', 'Medium')}`")

            with c_score:
                st.markdown(f"**Score:** `{item['score']}`")
                st.caption(f"Questions: `{item['total_questions']}`")

            with c_meta:
                st.caption(f"Completed: `{item['date']}`")

    st.markdown("---")

    col_clear, _ = st.columns([2, 3])
    with col_clear:
        if st.button("🗑️ Clear Quiz History", use_container_width=True):
            st.session_state.quiz_history = []
            st.success("Quiz history cleared!")
            st.rerun()

