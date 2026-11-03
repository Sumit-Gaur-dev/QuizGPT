"""
Session state initialization and management.
Provides default session variables and reset helpers for Streamlit.
"""

import uuid
import streamlit as st


def get_mock_questions(count: int = 5, difficulty: str = "Medium", question_types: list[str] = None) -> list[dict]:
    """Generate a clean set of mock questions for Phase 1/2/3 UI testing."""
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
            "question": "Which Python library is used for fast, native PDF text extraction in Quiz RAG?",
            "options": [
                "PyMuPDF (fitz)",
                "pdf2image",
                "ReportLab",
                "Selenium"
            ],
            "correct_answer": "PyMuPDF (fitz)",
            "explanation": "PyMuPDF provides high-speed text and page-level extraction without requiring external binary dependencies."
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

    filtered = [q for q in bank if q["type"] in question_types]
    if not filtered:
        filtered = bank

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

    # Uploaded file metadata list
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Extracted documents dictionary: {filename: extraction_result_dict}
    if "extracted_docs" not in st.session_state:
        st.session_state.extracted_docs = {}

    # Generated LangChain Document chunks list
    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    if "generated_quizzes_count" not in st.session_state:
        st.session_state.generated_quizzes_count = 0

    if "current_quiz" not in st.session_state:
        st.session_state.current_quiz = None

    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False

    if "quiz_history" not in st.session_state:
        st.session_state.quiz_history = []

    if "active_question_idx" not in st.session_state:
        st.session_state.active_question_idx = 0

    if "latest_evaluation" not in st.session_state:
        st.session_state.latest_evaluation = None

    if "force_mock_mode" not in st.session_state:
        st.session_state.force_mock_mode = True

    if "show_submit_confirm" not in st.session_state:
        st.session_state.show_submit_confirm = False


def reset_session_data() -> None:
    """Clear all session data and delete ChromaDB vector store for this session."""
    old_session_id = st.session_state.get("session_id")
    if old_session_id:
        try:
            from rag.vector_store import delete_session_vector_store
            delete_session_vector_store(old_session_id)
        except Exception:
            pass

    st.session_state.uploaded_files = []
    st.session_state.extracted_docs = {}
    st.session_state.chunks = []
    st.session_state.current_quiz = None
    st.session_state.user_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.active_question_idx = 0
    st.session_state.latest_evaluation = None
    st.session_state.show_submit_confirm = False
    st.session_state.generated_quizzes_count = 0
    st.session_state.quiz_history = []
    st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
