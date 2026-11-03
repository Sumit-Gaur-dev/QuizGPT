"""
Quiz Results Dashboard for Quiz RAG.
Renders comprehensive performance metrics, separates objective scoring from
AI-evaluated scoring, displays question-by-question evidence citations,
and provides constructive feedback.
"""

import streamlit as st


def render_quiz_results() -> None:
    """Render the comprehensive quiz results and grading breakdown."""
    evaluation = st.session_state.get("latest_evaluation")

    if not evaluation:
        st.info("No evaluation data found. Please complete and submit a quiz first.")
        if st.button("Go to Quiz Generator ➔", type="primary"):
            st.session_state.current_page = "⚙️ Generate Quiz"
            st.rerun()
        return

    st.header("🏆 Quiz Results & Evaluation")
    st.markdown(f"### {evaluation.get('title', 'Study Quiz')}")
    st.caption(
        f"Topic: **{evaluation.get('topic', 'General')}** | "
        f"Difficulty: **{evaluation.get('difficulty', 'Medium')}** | "
        f"Total Questions: **{evaluation.get('total_questions', 0)}**"
    )

    st.divider()

    # 1. Performance Metric Cards Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    total_obtained = evaluation.get("total_score_obtained", 0.0)
    total_possible = evaluation.get("total_score_possible", 0.0)
    pct = evaluation.get("percentage", 0.0)

    obj_obtained = evaluation.get("objective_score", 0.0)
    obj_max = evaluation.get("objective_max", 0.0)

    ai_obtained = evaluation.get("short_answer_score", 0.0)
    ai_max = evaluation.get("short_answer_max", 0.0)

    unanswered = evaluation.get("unanswered_questions", 0)

    with m_col1:
        st.metric(
            label="🎯 Overall Score",
            value=f"{pct}%",
            delta=f"{total_obtained} / {total_possible} pts"
        )

    with m_col2:
        st.metric(
            label="⚡ Objective Score",
            value=f"{obj_obtained} / {obj_max}",
            delta="Pure Python (Exact)" if obj_max > 0 else "N/A"
        )

    with m_col3:
        st.metric(
            label="🤖 Short Answer Score",
            value=f"{ai_obtained} / {ai_max}",
            delta="Rubric / AI Evaluated" if ai_max > 0 else "N/A"
        )

    with m_col4:
        st.metric(
            label="📝 Unanswered",
            value=f"{unanswered}",
            delta="0 pts awarded" if unanswered > 0 else "All answered!",
            delta_color="inverse" if unanswered > 0 else "normal"
        )

    st.divider()

    # 2. Detailed Question-by-Question Breakdown
    st.subheader("📋 Detailed Question Review & Citations")

    question_results = evaluation.get("question_results", [])

    for idx, qr in enumerate(question_results, start=1):
        q_type = qr.get("question_type", "MCQ")
        marks = qr.get("marks_obtained", 0.0)
        max_marks = qr.get("max_marks", 1.0)
        is_correct = qr.get("is_correct", False)
        is_unanswered = qr.get("is_unanswered", False)
        source_doc = qr.get("source_filename", "Notes.pdf")
        source_page = qr.get("source_page", 1)
        eval_type = qr.get("evaluation_type", "Standard")

        # Determine visual header styling
        if is_unanswered:
            status_icon = "⚠️"
            status_text = "UNANSWERED (0.0 pts)"
        elif is_correct or marks == max_marks:
            status_icon = "✅"
            status_text = f"CORRECT ({marks}/{max_marks} pts)"
        elif marks > 0.0:
            status_icon = "🟡"
            status_text = f"PARTIAL CREDIT ({marks}/{max_marks} pts)"
        else:
            status_icon = "❌"
            status_text = f"INCORRECT ({marks}/{max_marks} pts)"

        with st.container(border=True):
            col_hdr, col_score = st.columns([4, 1.5])
            with col_hdr:
                st.markdown(
                    f"**Question {idx}:** {qr.get('question')} "
                    f"`[{q_type}]`"
                )
            with col_score:
                st.markdown(f"**{status_icon} {status_text}**")
                st.caption(f"Evaluator: `{eval_type}`")

            st.write("")

            # User Answer vs Expected Answer
            c_ans1, c_ans2 = st.columns(2)
            with c_ans1:
                st.markdown(f"**Your Answer:**")
                ans_display = qr.get("user_answer", "(Not answered)")
                if is_unanswered:
                    st.warning(f"*(No answer was submitted)*")
                else:
                    st.info(f"{ans_display}")

            with c_ans2:
                st.markdown(f"**Expected / Reference Answer:**")
                st.success(f"{qr.get('correct_answer', 'N/A')}")

            # Specific feedback for Short Answer questions
            if q_type == "Short Answer":
                feedback = qr.get("feedback")
                missing = qr.get("missing_concepts", [])
                improvements = qr.get("suggested_improvements")

                st.markdown("---")
                st.markdown("💬 **AI Evaluator Feedback:**")
                if feedback:
                    st.write(f"- **Strengths:** {feedback}")
                if missing:
                    st.write(f"- **Missing Concepts:** `{', '.join(missing)}`")
                if improvements:
                    st.write(f"- **Key Improvement:** {improvements}")

            # Grounding and citation information
            st.markdown("---")
            col_cite, col_exp = st.columns([1.5, 2.5])
            with col_cite:
                st.markdown(f"📄 **Source Citation:**")
                st.caption(f"Document: **`{source_doc}`**")
                st.caption(f"Page Number: **Page {source_page}**")
            with col_exp:
                st.markdown(f"💡 **Explanation from Study Notes:**")
                st.caption(f"{qr.get('explanation', 'No explanation provided.')}")

    st.divider()

    # 3. Action Buttons
    col_act1, col_act2, col_act3 = st.columns(3)

    with col_act1:
        if st.button("🔄 Retake This Quiz", use_container_width=True):
            st.session_state.user_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.active_question_idx = 0
            st.session_state.latest_evaluation = None
            st.session_state.show_submit_confirm = False
            st.rerun()

    with col_act2:
        if st.button("⚙️ Generate a New Quiz", type="primary", use_container_width=True):
            st.session_state.current_page = "⚙️ Generate Quiz"
            st.rerun()

    with col_act3:
        if st.button("📜 View Quiz History", use_container_width=True):
            st.session_state.current_page = "📜 Quiz History"
            st.rerun()
