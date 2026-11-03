"""
Evaluation and Scoring Module for Quiz RAG.
Provides deterministic Python evaluation for objective questions (MCQ, True/False)
and structured rubric-based evaluation for short-answer questions.
Strictly decoupled from Streamlit UI logic.
"""

from typing import Any
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate

from config.settings import has_valid_gemini_key
from rag.generation import get_gemini_chat_model


# =====================================================================
# Pydantic Schema for Short Answer Evaluation
# =====================================================================

class ShortAnswerEvaluation(BaseModel):
    """Pydantic schema representing the AI evaluation of a short answer response."""
    marks_obtained: float = Field(
        ge=0.0,
        le=5.0,
        description="Marks awarded from 0.0 to 5.0 based on conceptual completeness and accuracy."
    )
    max_marks: float = Field(
        default=5.0,
        description="Maximum possible marks for this question."
    )
    feedback: str = Field(
        description="Constructive assessment explaining the strengths of the answer."
    )
    missing_concepts: list[str] = Field(
        default_factory=list,
        description="Key ideas, terminology, or concepts required by the notes that the user omitted."
    )
    suggested_improvements: str = Field(
        description="Specific actionable advice on how to improve the response."
    )

    @field_validator("marks_obtained")
    def clamp_marks(cls, v: float, values: Any) -> float:
        """Guarantee marks stay strictly within valid bounds."""
        return max(0.0, min(5.0, round(float(v), 1)))


# =====================================================================
# Evaluation Prompts for Short Answers
# =====================================================================

EVAL_SYSTEM_PROMPT = """You are an objective academic evaluator assessing a student's short answer against study notes.

EVALUATION RUBRIC (Max 5.0 Marks):
- 5.0 Marks: Exceptional, accurate, covers all key concepts thoroughly with correct terminology.
- 3.5 - 4.5 Marks: Good understanding, covers most core concepts, minor details missing.
- 2.0 - 3.0 Marks: Partially correct, captures general idea but misses critical concepts.
- 0.5 - 1.5 Marks: Minimal understanding, vague or largely off-topic.
- 0.0 Marks: Blank, irrelevant, or factually contradicted by the source context.

GROUNDING RULES:
1. Do NOT award marks for assertions that contradict the supplied SOURCE CONTEXT.
2. Evaluate strictly against the supplied REFERENCE ANSWER and KEY CONCEPTS.
3. Return your assessment in the structured schema.
"""

EVAL_USER_PROMPT = """Evaluate this student answer:

QUESTION:
{question}

SOURCE CONTEXT:
{context}

EXPECTED REFERENCE ANSWER:
{reference_answer}

KEY CONCEPTS REQUIRED:
{key_concepts}

STUDENT'S ANSWER:
{student_answer}
"""


# =====================================================================
# Objective Evaluation (Pure Python — Zero LLM Calls)
# =====================================================================

def evaluate_objective_question(question: dict[str, Any], user_answer: str | None) -> dict[str, Any]:
    """
    Deterministically evaluate an MCQ or True/False question using exact normalized string matching.
    Zero API calls or LLM dependencies.
    """
    correct_raw = str(question.get("correct_answer", "")).strip()
    is_unanswered = not user_answer or not str(user_answer).strip()

    if is_unanswered:
        is_correct = False
    else:
        user_clean = str(user_answer).strip().lower()
        correct_clean = correct_raw.lower()
        is_correct = (user_clean == correct_clean)

    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "question_type": question["question_type"],
        "difficulty": question.get("difficulty", "Medium"),
        "user_answer": user_answer if not is_unanswered else "(Not Answered)",
        "correct_answer": correct_raw,
        "is_correct": is_correct,
        "is_unanswered": is_unanswered,
        "explanation": question.get("explanation", ""),
        "source_filename": question.get("source_filename", "Notes.pdf"),
        "source_page": question.get("source_page", 1),
        "marks_obtained": 1.0 if is_correct else 0.0,
        "max_marks": 1.0,
        "evaluation_type": "Deterministic (Python)"
    }


# =====================================================================
# Short Answer Evaluation (Gemini or Verified Offline Mock)
# =====================================================================

def evaluate_short_answer_mock(question: dict[str, Any], user_answer: str | None) -> ShortAnswerEvaluation:
    """
    Offline rubric-based evaluator using keyword matching against key concepts.
    Used during development when no API key is set or when mock mode is enabled.
    Zero external network calls.
    """
    if not user_answer or not str(user_answer).strip():
        return ShortAnswerEvaluation(
            marks_obtained=0.0,
            max_marks=5.0,
            feedback="No answer was provided for this question.",
            missing_concepts=question.get("key_concepts", []),
            suggested_improvements="Review the lecture notes on this page and attempt an explanation."
        )

    user_text = str(user_answer).strip().lower()
    key_concepts = question.get("key_concepts", [])
    if not key_concepts:
        key_concepts = [question.get("reference_answer", "")]

    # Calculate matched concepts
    matched = [c for c in key_concepts if any(w in user_text for w in c.lower().split() if len(w) > 3)]
    missing = [c for c in key_concepts if c not in matched]

    ratio = len(matched) / max(1, len(key_concepts))
    marks = round(max(1.0, min(5.0, ratio * 5.0)), 1)

    if ratio >= 0.8:
        feedback = "Excellent response! Your explanation covers the essential key concepts clearly."
        improvements = "To achieve full marks, ensure exact domain terminology is included."
    elif ratio >= 0.5:
        feedback = "Good effort. You captured several important concepts, but omitted some key details."
        improvements = f"Make sure to specifically mention: {', '.join(missing)}."
    else:
        feedback = "Partial understanding shown. The response needs more alignment with the study notes."
        improvements = f"Refer to the source notes to address: {', '.join(missing)}."

    return ShortAnswerEvaluation(
        marks_obtained=marks,
        max_marks=5.0,
        feedback=feedback,
        missing_concepts=missing,
        suggested_improvements=improvements
    )


def evaluate_short_answer_live(question: dict[str, Any], user_answer: str | None) -> ShortAnswerEvaluation:
    """
    Evaluate a short answer question live using ChatGoogleGenerativeAI with structured output.
    Only called when a valid GOOGLE_API_KEY is configured.
    """
    # Blank answer guardrail: zero marks without calling the LLM
    if not user_answer or not str(user_answer).strip():
        return ShortAnswerEvaluation(
            marks_obtained=0.0,
            max_marks=5.0,
            feedback="No answer was submitted for this question.",
            missing_concepts=question.get("key_concepts", []),
            suggested_improvements="Review the reference answer and notes before attempting again."
        )

    llm = get_gemini_chat_model()
    structured_evaluator = llm.with_structured_output(ShortAnswerEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVAL_SYSTEM_PROMPT),
        ("human", EVAL_USER_PROMPT),
    ])

    chain = prompt | structured_evaluator

    eval_result: ShortAnswerEvaluation = chain.invoke({
        "question": question["question"],
        "context": question.get("explanation", ""),
        "reference_answer": question.get("reference_answer", ""),
        "key_concepts": ", ".join(question.get("key_concepts", [])),
        "student_answer": user_answer
    })

    return eval_result


def evaluate_short_answer_question(
    question: dict[str, Any],
    user_answer: str | None,
    force_mock: bool = False
) -> dict[str, Any]:
    """
    Evaluate a short-answer question, choosing live Gemini evaluation if an API key is present,
    or the offline rubric mock during development.
    """
    is_unanswered = not user_answer or not str(user_answer).strip()

    if force_mock or not has_valid_gemini_key():
        eval_model = evaluate_short_answer_mock(question, user_answer)
        eval_type = "Offline Rubric (Mock AI)"
    else:
        eval_model = evaluate_short_answer_live(question, user_answer)
        eval_type = "Gemini LLM (Live AI)"

    return {
        "question_id": question["question_id"],
        "question": question["question"],
        "question_type": "Short Answer",
        "difficulty": question.get("difficulty", "Medium"),
        "user_answer": user_answer if not is_unanswered else "(Not Answered)",
        "correct_answer": question.get("reference_answer", ""),
        "is_correct": eval_model.marks_obtained >= 3.0,
        "is_unanswered": is_unanswered,
        "explanation": question.get("explanation", ""),
        "source_filename": question.get("source_filename", "Notes.pdf"),
        "source_page": question.get("source_page", 1),
        "marks_obtained": eval_model.marks_obtained,
        "max_marks": eval_model.max_marks,
        "feedback": eval_model.feedback,
        "missing_concepts": eval_model.missing_concepts,
        "suggested_improvements": eval_model.suggested_improvements,
        "evaluation_type": eval_type
    }


# =====================================================================
# Complete Quiz Evaluation Coordinator
# =====================================================================

def evaluate_complete_quiz(
    quiz: dict[str, Any],
    user_answers: dict[int | str, Any],
    force_mock: bool = False
) -> dict[str, Any]:
    """
    Evaluate all answers in a submitted quiz.
    Separates deterministic objective scoring from AI-assisted short answer scoring.
    
    Args:
        quiz: Quiz dictionary containing 'questions' list.
        user_answers: Dictionary mapping question_id -> user answer string.
        force_mock: If True, uses offline mock evaluation for short answers.
        
    Returns:
        Structured results summary dictionary.
    """
    questions = quiz.get("questions", [])
    results: list[dict[str, Any]] = []

    objective_score = 0.0
    objective_max = 0.0
    short_answer_score = 0.0
    short_answer_max = 0.0

    correct_count = 0
    incorrect_count = 0
    unanswered_count = 0

    for q in questions:
        q_id = q["question_id"]
        # Handle string or int keys from session state
        ans = user_answers.get(q_id)
        if ans is None:
            ans = user_answers.get(str(q_id))

        q_type = q.get("question_type", "MCQ")

        if q_type in ("MCQ", "True/False"):
            q_res = evaluate_objective_question(q, ans)
            objective_score += q_res["marks_obtained"]
            objective_max += q_res["max_marks"]
        else: # Short Answer
            q_res = evaluate_short_answer_question(q, ans, force_mock=force_mock)
            short_answer_score += q_res["marks_obtained"]
            short_answer_max += q_res["max_marks"]

        if q_res["is_unanswered"]:
            unanswered_count += 1
        elif q_res["is_correct"]:
            correct_count += 1
        else:
            incorrect_count += 1

        results.append(q_res)

    total_obtained = round(objective_score + short_answer_score, 1)
    total_possible = round(objective_max + short_answer_max, 1)
    percentage = round((total_obtained / total_possible * 100), 1) if total_possible > 0 else 0.0

    return {
        "title": quiz.get("title", "Study Quiz"),
        "topic": quiz.get("topic", "General Notes"),
        "difficulty": quiz.get("difficulty", "Medium"),
        "total_questions": len(questions),
        "correct_answers": correct_count,
        "incorrect_answers": incorrect_count,
        "unanswered_questions": unanswered_count,
        "objective_score": round(objective_score, 1),
        "objective_max": round(objective_max, 1),
        "short_answer_score": round(short_answer_score, 1),
        "short_answer_max": round(short_answer_max, 1),
        "total_score_obtained": total_obtained,
        "total_score_possible": total_possible,
        "percentage": percentage,
        "question_results": results
    }

