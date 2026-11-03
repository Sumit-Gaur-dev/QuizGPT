"""
Quiz attempt view for Quiz RAG.
Features question-by-question navigation, real-time progress tracking,
answered/unanswered status badges, submission confirmation, and delegates to
the Results Dashboard upon grading.
"""

from datetime import datetime
import streamlit as st
from frontend.quiz_results import render_quiz_results
from rag.evaluation import evaluate_complete_quiz


def render_quiz_attempt() -> None:
    """Render the interactive question-by-question quiz taking view."""
    # 1. If quiz has already been submitted and graded, show results
    if st.session_state.get("quiz_submitted", False):
        render_quiz_results()
        return

    quiz = st.session_state.get("current_quiz")

    if not quiz or not quiz.get("questions"):
        st.info("No active quiz available. Please configure and generate a quiz first.")
        if st.button("Go to Quiz Generator ➔", type="primary"):
            st.session_state.current_page = "⚙️ Generate Quiz"
            st.rerun()
        return

    questions = quiz["questions"]
    total_q = len(questions)

    # Ensure active question index is within range
    if "active_question_idx" not in st.session_state:
        st.session_state.active_question_idx = 0
    if st.session_state.active_question_idx >= total_q:
        st.session_state.active_question_idx = total_q - 1
    if st.session_state.active_question_idx < 0:
        st.session_state.active_question_idx = 0

    curr_idx = st.session_state.active_question_idx
    curr_q = questions[curr_idx]
    q_id = curr_q.get("question_id", curr_idx + 1)

    # 2. Header and Meta Info
    st.header(f"✍️ {quiz.get('title', 'Study Notes Quiz')}")
    col_top1, col_top2, col_top3, col_top4 = st.columns(4)
    with col_top1:
        st.caption(f"Topic: **{quiz.get('topic', 'General')}**")
    with col_top2:
        st.caption(f"Difficulty: **{quiz.get('difficulty', 'Medium')}**")
    with col_top3:
        st.caption(f"Questions: **{total_q} total**")
    with col_top4:
        mode_badge = "Offline Mock" if st.session_state.get("force_mock_mode", True) else "Live Gemini"
        st.caption(f"Mode: **{mode_badge}**")

    st.divider()

    # 3. Progress Tracking & Counter
    # Count how many questions currently have non-empty answers
    answered_count = 0
    for q in questions:
        qid = q.get("question_id", 0)
        val = st.session_state.user_answers.get(qid) or st.session_state.user_answers.get(str(qid))
        if val and str(val).strip():
            answered_count += 1

    progress_ratio = answered_count / total_q if total_q > 0 else 0.0

    col_prog, col_badge = st.columns([3, 1])
    with col_prog:
        st.progress(progress_ratio)
        st.caption(f"Question **{curr_idx + 1}** of **{total_q}** ({int(progress_ratio * 100)}% answered)")
    with col_badge:
        st.markdown(
            f"**Answered:** `{answered_count}/{total_q}` &nbsp;|&nbsp; "
            f"**Pending:** `{total_q - answered_count}`"
        )

    # 4. Direct Jump Navigation Buttons
    st.write("")
    jump_cols = st.columns(total_q)
    for i, q in enumerate(questions):
        qid = q.get("question_id", i + 1)
        val = st.session_state.user_answers.get(qid) or st.session_state.user_answers.get(str(qid))
        has_ans = bool(val and str(val).strip())

        # Label styling: checkmark if answered, arrow if currently selected
        if i == curr_idx:
            btn_label = f"▶ Q{i+1}"
        elif has_ans:
            btn_label = f"✓ Q{i+1}"
        else:
            btn_label = f"Q{i+1}"

        with jump_cols[i]:
            if st.button(
                btn_label,
                key=f"jump_btn_{i}",
                use_container_width=True,
                type="primary" if i == curr_idx else "secondary"
            ):
                st.session_state.active_question_idx = i
                st.rerun()

    st.write("")

    # 5. Question Interaction Card
    with st.container(border=True):
        col_qhdr, col_type = st.columns([4, 1.2])
        with col_qhdr:
            st.markdown(f"### Question {curr_idx + 1}")
        with col_type:
            st.info(f"Format: **{curr_q.get('question_type', 'MCQ')}**")

        st.markdown(f"#### {curr_q.get('question')}")
        st.write("")

        q_type = curr_q.get("question_type", "MCQ")
        saved_ans = st.session_state.user_answers.get(q_id) or st.session_state.user_answers.get(str(q_id))

        if q_type == "MCQ":
            options = curr_q.get("options", [])
            default_index = options.index(saved_ans) if saved_ans in options else None
            selected = st.radio(
                label=f"Choose the best answer for Question {curr_idx + 1}:",
                options=options,
                index=default_index,
                key=f"attempt_mcq_{q_id}"
            )
            if selected:
                st.session_state.user_answers[q_id] = selected

        elif q_type == "True/False":
            options = ["True", "False"]
            default_index = options.index(saved_ans) if saved_ans in options else None
            selected = st.radio(
                label=f"Choose True or False for Question {curr_idx + 1}:",
                options=options,
                index=default_index,
                key=f"attempt_tf_{q_id}"
            )
            if selected:
                st.session_state.user_answers[q_id] = selected

        else: # Short Answer
            user_text = st.text_area(
                label=f"Write your concise answer for Question {curr_idx + 1}:",
                value=saved_ans if saved_ans else "",
                placeholder="Explain the concept based on your study notes...",
                key=f"attempt_sa_{q_id}",
                height=120
            )
            if user_text:
                st.session_state.user_answers[q_id] = user_text

    # 6. Previous / Next Navigation Controls
    col_prev, col_spacer, col_next = st.columns([1.5, 3, 1.5])

    with col_prev:
        if st.button("⬅️ Previous Question", disabled=(curr_idx == 0), use_container_width=True):
            st.session_state.active_question_idx -= 1
            st.rerun()

    with col_next:
        if st.button("Next Question ➡️", disabled=(curr_idx == total_q - 1), use_container_width=True):
            st.session_state.active_question_idx += 1
            st.rerun()

    st.markdown("---")

    # 7. Quiz Submission & Confirmation Dialog
    if not st.session_state.get("show_submit_confirm", False):
        col_sub_btn, _ = st.columns([2, 3])
        with col_sub_btn:
            if st.button("📨 Submit Quiz for Evaluation", type="primary", use_container_width=True):
                st.session_state.show_submit_confirm = True
                st.rerun()
    else:
        with st.container(border=True):
            st.markdown("### ⚠️ Confirm Quiz Submission")
            unanswered = total_q - answered_count

            if unanswered > 0:
                st.warning(
                    f"You have answered **{answered_count} of {total_q}** questions. "
                    f"**{unanswered} question(s) remain unanswered** and will receive 0 marks."
                )
            else:
                st.success(f"Great job! You have answered all **{total_q} of {total_q}** questions.")

            st.write("Are you sure you want to finish and evaluate your quiz?")

            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("✅ Yes, Submit & Grade Now", type="primary", use_container_width=True):
                    with st.spinner("Evaluating your responses against grounded notes..."):
                        evaluation = evaluate_complete_quiz(
                            quiz=quiz,
                            user_answers=st.session_state.user_answers,
                            force_mock=st.session_state.get("force_mock_mode", True)
                        )

                        st.session_state.latest_evaluation = evaluation
                        st.session_state.quiz_submitted = True
                        st.session_state.show_submit_confirm = False

                        # Record in quiz history
                        mode_label = "Offline Mock" if st.session_state.get("force_mock_mode", True) else "Live Gemini"
                        st.session_state.quiz_history.insert(0, {
                            "id": f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            "title": quiz.get("title", "Study Quiz"),
                            "topic": quiz.get("topic", "General"),
                            "difficulty": quiz.get("difficulty", "Medium"),
                            "total_questions": total_q,
                            "score": f"{evaluation.get('total_score_obtained', 0)} / {evaluation.get('total_score_possible', 0)}",
                            "percentage": evaluation.get("percentage", 0),
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "mode": mode_label,
                            "evaluation": evaluation
                        })

                    st.rerun()

            with c_no:
                if st.button("❌ Cancel / Keep Working", use_container_width=True):
                    st.session_state.show_submit_confirm = False
                    st.rerun()
