"""
Quiz history view for Quiz RAG.
Displays previous quiz attempts, score breakdowns, and completion history.
"""

import streamlit as st


def render_quiz_history() -> None:
    """Render the quiz history and past attempts summary."""
    st.header("📜 Quiz History & Past Attempts")
    st.markdown(
        "View records of all your generated and completed quizzes from this session. "
        "Scores and detailed evaluations are recorded in session state."
    )

    st.divider()

    history = st.session_state.get("quiz_history", [])

    if not history:
        st.info("No quiz attempts recorded yet. Generate and complete a quiz to see it listed here.")
        if st.button("Go to Quiz Generator ➔", type="primary"):
            st.session_state.current_page = "⚙️ Generate Quiz"
            st.rerun()
        return

    # Metric summaries
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Quizzes Completed", value=len(history))
    with col2:
        latest = history[0]
        st.metric(
            label="Latest Score",
            value=f"{latest.get('score', 'N/A')}",
            delta=f"{latest.get('percentage', 0)}%" if 'percentage' in latest else None
        )
    with col3:
        avg_pct = round(sum(h.get("percentage", 0) for h in history) / len(history), 1)
        st.metric(label="Average Performance", value=f"{avg_pct}%")

    st.divider()

    st.subheader("📋 Completed Quizzes")

    for idx, item in enumerate(history):
        with st.container(border=True):
            c_title, c_score, c_meta, c_action = st.columns([3, 2, 2, 1.5])

            with c_title:
                st.markdown(f"🎯 **{item['title']}**")
                st.caption(f"Topic: `{item.get('topic', 'General')}` | Difficulty: `{item.get('difficulty', 'Medium')}`")

            with c_score:
                st.markdown(f"**Score:** `{item['score']}` ({item.get('percentage', 0)}%)")
                st.caption(f"Questions: `{item['total_questions']}`")

            with c_meta:
                st.caption(f"Completed: `{item['date']}`")
                mode_str = item.get("mode", "Offline Mock")
                st.caption(f"Evaluator: `{mode_str}`")

            with c_action:
                # If this item has full evaluation stored, allow reviewing it
                if item.get("evaluation"):
                    if st.button("Review ➔", key=f"review_hist_{idx}", use_container_width=True):
                        st.session_state.latest_evaluation = item["evaluation"]
                        st.session_state.quiz_submitted = True
                        st.session_state.current_page = "✍️ Attempt Quiz"
                        st.rerun()

    st.markdown("---")

    col_clear, _ = st.columns([2, 3])
    with col_clear:
        if st.button("🗑️ Clear Quiz History", use_container_width=True):
            st.session_state.quiz_history = []
            st.toast("Quiz history cleared!", icon="🧹")
            st.rerun()

