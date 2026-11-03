"""
Session state initialization and management.
Provides default session variables and reset helpers for Streamlit.
"""

import uuid
import streamlit as st


def get_mock_questions(count: int = 5, difficulty: str = "Medium", question_types: list[str] = None) -> list[dict]:
    """Generate a clean set of mock questions for Phase 1 UI testing."""
    if question_types is None:
        question_types = ["MCQ", "True/False", "Short Answer"]

    bank = [
        {
            "id": 1,
            "type": "MCQ",
            "question": "What is the primary role of an embedding model in a RAG pipeline?",
            "options": [
                "Converting text chunks into dense numerical vector representations",
                "Rendering frontend HTML templates inside the browser",
                "Running full-text regex searches on raw PDF files",
                "Encrypting user passwords before database storage"
            ],
            "correct_answer": "Converting text chunks into dense numerical vector representations",
            "explanation": "Embedding models map semantic text into vector space so similarity search can retrieve relevant context."
        },
        {
            "id": 2,
            "type": "True/False",
            "question": "ChromaDB requires an external cloud subscription and cannot run locally in Python.",
            "options": ["True", "False"],
            "correct_answer": "False",
            "explanation": "ChromaDB runs completely locally using PersistentClient or EphemeralClient without needing any cloud service."
        },
        {
            "id": 3,
            "type": "Short Answer",
            "question": "Why is chunk overlap used when splitting documents for RAG retrieval?",
            "correct_answer": "Chunk overlap prevents splitting concepts across boundaries and maintains contextual continuity.",
            "explanation": "Overlap guarantees that statements occurring near a split boundary remain intact and retrievable."
        },
        {
            "id": 4,
            "type": "MCQ",
            "question": "Which Python library will be used in Phase 2 for fast, native PDF text extraction?",
            "options": [
                "PyMuPDF (fitz)",
                "pdf2image",
                "ReportLab",
                "Selenium"
            ],
            "correct_answer": "PyMuPDF (fitz)",
            "explanation": "PyMuPDF provides high-speed text and page-level extraction without requiring heavy external binaries."
        },
        {
            "id": 5,
            "type": "True/False",
            "question": "Objective questions (like MCQs and True/False) should be evaluated deterministically in Python rather than calling an LLM.",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": "Deterministic string comparison in Python is 100% accurate, instantaneous, and avoids unnecessary API costs."
        },
        {
            "id": 6,
            "type": "MCQ",
            "question": "What is the danger of using a single shared ChromaDB collection for multiple concurrent visitors?",
            "options": [
                "One visitor's search queries can retrieve another visitor's private notes",
                "Streamlit will crash due to thread locks",
                "The embedding dimension automatically resets to zero",
                "PDF files will be permanently deleted from the hard drive"
            ],
            "correct_answer": "One visitor's search queries can retrieve another visitor's private notes",
            "explanation": "Without session-isolated collections, data from multiple users gets mixed together in similarity search."
        },
        {
            "id": 7,
            "type": "Short Answer",
            "question": "What is 'strict grounding' in the context of LLM quiz generation?",
            "correct_answer": "Instructing the LLM to generate questions and answers solely from the retrieved context without hallucinating outside facts.",
            "explanation": "Strict grounding ensures quizzes test material actually found in the uploaded study notes."
        }
    ]

    # Filter bank by chosen question types if possible
    filtered = [q for q in bank if q["type"] in question_types]
    if not filtered:
        filtered = bank

    # Repeat or slice to match requested count
    questions = []
    idx = 0
    for i in range(count):
        base_q = filtered[idx % len(filtered)].copy()
        base_q["id"] = i + 1
        base_q["difficulty"] = difficulty
        questions.append(base_q)
        idx += 1

    return questions


def init_session_state() -> None:
    """Initialize default keys in st.session_state if they do not exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

    if "current_page" not in st.session_state:
        st.session_state.current_page = "📊 Dashboard"

    if "uploaded_files" not in st.session_state:
        # Prepopulate with 1 mock sample note so the user can test immediately
        st.session_state.uploaded_files = [
            {
                "id": "sample_note_1",
                "name": "Lecture_01_RAG_Fundamentals.pdf",
                "size_kb": 245.5,
                "pages": 12,
                "status": "Ready for Indexing (Mock)"
            }
        ]

    if "generated_quizzes_count" not in st.session_state:
        st.session_state.generated_quizzes_count = 2

    if "current_quiz" not in st.session_state:
        # Initial active mock quiz
        st.session_state.current_quiz = {
            "title": "RAG Concepts & Architecture Quiz",
            "topic": "Retrieval-Augmented Generation Basics",
            "difficulty": "Medium",
            "total_questions": 5,
            "questions": get_mock_questions(count=5, difficulty="Medium")
        }

    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    if "quiz_history" not in st.session_state:
        # Prepopulate sample completed quizzes for history inspection
        st.session_state.quiz_history = [
            {
                "id": "hist_1",
                "title": "Python Data Structures Primer",
                "topic": "Lists, Dictionaries & Tuples",
                "difficulty": "Easy",
                "total_questions": 5,
                "score": "5 / 5 (100%)",
                "date": "2026-11-02 11:30"
            },
            {
                "id": "hist_2",
                "title": "Vector Databases Overview",
                "topic": "ChromaDB & Similarity Search",
                "difficulty": "Medium",
                "total_questions": 5,
                "score": "4 / 5 (80%)",
                "date": "2026-11-03 09:15"
            }
        ]


def reset_session_data() -> None:
    """Clear all session data (implements the 'Delete my data' action)."""
    st.session_state.uploaded_files = []
    st.session_state.current_quiz = None
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.generated_quizzes_count = 0
    st.session_state.quiz_history = []
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"

