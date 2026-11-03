"""
Quiz attempt view for Quiz RAG.
Renders question forms, captures user responses, and submits attempts (Phase 1 mock).
"""

from datetime import datetime
import streamlit as st


def render_quiz_attempt() -> None:
    """Render the interactive quiz taking and mock submission view."""
    st.header("✍️ Quiz Attempt")

    quiz = st.session_state.get("current_quiz")

    if not quiz:
        st.info("No active quiz available. Please configure and generate a quiz first.")
        if st.button("Go to Quiz Generator ➔", type="primary"):
            st.session_state.current_page = "⚙️ Generate Quiz"
            st.rerun()
        return

    # Header info
    st.markdown(f"### {quiz['title']}")
    col_d, col_c, col_t = st.columns(3)
    with col_d:
        st.caption(f"Difficulty: **{quiz.get('difficulty', 'Medium')}**")
    with col_c:
        st.caption(f"Questions: **{len(quiz['questions'])}**")
    with col_t:
        st.caption(f"Topic: **{quiz.get('topic', 'General')}**")

    st.divider()

    # If already submitted, display the submission summary and review
    if st.session_state.get("quiz_submitted", False):
        st.success("✅ **Quiz Submitted!** (Phase 1 Mock Submission)")
        st.info(
            "In Phase 1, actual scoring and Gemini short-answer evaluation are mocked. "
            "Real deterministic scoring for MCQs and AI-assisted rubric grading will activate in Phase 6."
        )

        st.subheader("📋 Attempt Review")
        for q in quiz["questions"]:
            q_id = q["id"]
            user_ans = st.session_state.user_answers.get(q_id, "*(Not answered)*")

            with st.container(border=True):
                st.markdown(f"**Q{q_id} ({q['type']}):** {q['question']}")
                st.markdown(f"- **Your Answer:** `{user_ans}`")
                st.markdown(f"- **Expected Answer (Mock):** `{q.get('correct_answer', 'N/A')}`")
                st.caption(f"💡 *Explanation:* {q.get('explanation', '')}")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake This Quiz", use_container_width=True):
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()
        with col2:
            if st.button("⚙️ Generate a New Quiz", type="primary", use_container_width=True):
                st.session_state.current_page = "⚙️ Generate Quiz"
                st.rerun()
        return

    # Active quiz taking form
    st.markdown("Select or write your answers below. Answers are preserved in `st.session_state`.")

    with st.form("quiz_attempt_form"):
        for q in quiz["questions"]:
            q_id = q["id"]
            st.markdown(f"#### Question {q_id} <span style='font-size:0.8em; color:gray;'>({q['type']})</span>", unsafe_allow_html=True)
            st.write(q["question"])

            current_val = st.session_state.user_answers.get(q_id)

            if q["type"] == "MCQ":
                opts = q.get("options", [])
                default_idx = opts.index(current_val) if current_val in opts else None
                selected = st.radio(
                    label=f"Choose an option for Q{q_id}:",
                    options=opts,
                    index=default_idx,
                    key=f"q_radio_{q_id}",
                    label_visibility="collapsed"
                )
                if selected:
                    st.session_state.user_answers[q_id] = selected

            elif q["type"] == "True/False":
                opts = ["True", "False"]
                default_idx = opts.index(current_val) if current_val in opts else None
                selected = st.radio(
                    label=f"Choose True or False for Q{q_id}:",
                    options=opts,
                    index=default_idx,
                    key=f"q_tf_{q_id}",
                    label_visibility="collapsed"
                )
                if selected:
                    st.session_state.user_answers[q_id] = selected

            elif q["type"] == "Short Answer":
                answer_text = st.text_area(
                    label=f"Write your short answer for Q{q_id}:",
                    value=current_val if current_val else "",
                    placeholder="Type your concise answer here...",
                    key=f"q_text_{q_id}",
                    label_visibility="collapsed"
                )
                if answer_text:
                    st.session_state.user_answers[q_id] = answer_text

            st.write("")

        submit_btn = st.form_submit_button("📨 Submit Quiz", type="primary", use_container_width=True)

    if submit_btn:
        st.session_state.quiz_submitted = True

        # Calculate a mock score for Phase 1 demo
        total_q = len(quiz["questions"])
        answered_q = len(st.session_state.user_answers)
        mock_score = f"{max(1, answered_q)} / {total_q} (Mock Score)"

        # Record into quiz history
        st.session_state.quiz_history.insert(0, {
            "id": f"hist_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": quiz["title"],
            "topic": quiz.get("topic", "General"),
            "difficulty": quiz.get("difficulty", "Medium"),
            "total_questions": total_q,
            "score": mock_score,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

        st.rerun()

