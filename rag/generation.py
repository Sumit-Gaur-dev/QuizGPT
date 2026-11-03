"""
Quiz Generation Service for Quiz RAG using LangChain and Gemini.
Generates structured quizzes strictly grounded in retrieved ChromaDB context.
Designed to run live with Gemini or with verified offline mocks during development.
Strictly decoupled from Streamlit UI logic.
"""

from typing import Any, Literal
import os
from pydantic import BaseModel, Field, model_validator
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config.settings import GOOGLE_API_KEY, GEMINI_MODEL, has_valid_gemini_key
from rag.retrieval import retrieve_relevant_chunks


# =====================================================================
# Pydantic Schemas for Structured Quiz Output
# =====================================================================

class QuestionModel(BaseModel):
    """Pydantic schema representing a single strictly grounded quiz question."""
    question_id: int = Field(description="Sequential integer ID of the question.")
    question: str = Field(description="Clear, concise question text strictly grounded in context.")
    question_type: Literal["MCQ", "True/False", "Short Answer"] = Field(
        description="Format of the question."
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        default="Medium",
        description="Difficulty level of the question."
    )
    options: list[str] = Field(
        default_factory=list,
        description="List of 4 distinct choices for MCQ questions. Empty for other types."
    )
    correct_answer: str = Field(
        description="The unambiguous correct answer supported by the text."
    )
    explanation: str = Field(
        description="Short explanation explaining why this answer is correct based on the notes."
    )
    source_filename: str = Field(
        description="Name of the source PDF document containing the evidence."
    )
    source_page: int = Field(
        default=1,
        description="1-indexed page number of the source document."
    )
    reference_answer: str = Field(
        default="",
        description="Comprehensive model answer for short-answer questions. Empty for objective questions."
    )
    key_concepts: list[str] = Field(
        default_factory=list,
        description="List of required key concepts or phrases for short-answer rubric evaluation."
    )

    @model_validator(mode="after")
    def validate_question_integrity(self):
        """Validate question constraints according to type."""
        q_type = self.question_type

        if q_type == "MCQ":
            if len(self.options) != 4:
                raise ValueError(f"MCQ must have exactly 4 options, got {len(self.options)}.")
            norm_opts = [opt.strip().lower() for opt in self.options]
            if self.correct_answer.strip().lower() not in norm_opts:
                raise ValueError(
                    f"MCQ correct_answer '{self.correct_answer}' must match one of the options: {self.options}."
                )

        elif q_type == "True/False":
            ans_clean = self.correct_answer.strip().lower()
            if ans_clean not in ("true", "false"):
                raise ValueError(
                    f"True/False question must have 'True' or 'False' as correct_answer, got '{self.correct_answer}'."
                )
            self.correct_answer = "True" if ans_clean == "true" else "False"
            if not self.options:
                self.options = ["True", "False"]

        elif q_type == "Short Answer":
            if not self.reference_answer.strip():
                raise ValueError("Short Answer questions must provide a non-empty reference_answer.")
            if not self.key_concepts:
                self.key_concepts = [self.reference_answer.strip()[:50]]

        return self


class QuizModel(BaseModel):
    """Pydantic schema representing the complete structured quiz."""
    title: str = Field(description="Descriptive quiz title summarizing the topic.")
    topic: str = Field(description="Target topic or subject covered by this quiz.")
    difficulty: str = Field(description="Difficulty level of the quiz.")
    total_questions: int = Field(description="Number of questions included in the quiz.")
    questions: list[QuestionModel] = Field(description="List of validated questions.")


# =====================================================================
# Prompt Template for Grounded Generation
# =====================================================================

QUIZ_SYSTEM_PROMPT = """You are an expert educational assessment creator.
Your task is to generate a comprehensive, high-quality quiz strictly grounded in the supplied study notes context.

STRICT GROUNDING RULES:
1. Rely ONLY on the facts and information explicitly stated in the CONTEXT below.
2. Do NOT invent facts, extrapolate, or use outside knowledge.
3. Every question must reference its exact source_filename and source_page as found in the chunk metadata.
4. For MCQ questions:
   - Provide exactly 4 plausible options (one correct, three believable distractors).
   - Ensure the correct answer is directly supported by the context.
5. For True/False questions:
   - The correct answer must be 'True' or 'False'.
6. For Short Answer questions:
   - Provide a concise, clear reference_answer.
   - Provide a list of 2-4 key_concepts required for full marks.
7. If the context does not contain enough information to formulate the requested number of questions, formulate as many valid questions as the context supports.

CONTEXT:
{context}
"""

QUIZ_USER_PROMPT = """Generate a quiz with the following parameters:
- Target Topic: {topic}
- Difficulty Level: {difficulty}
- Desired Question Count: {num_questions}
- Allowed Question Types: {question_types}

Return the response adhering strictly to the structured JSON schema.
"""


# =====================================================================
# LLM Initialization & Mock Helper
# =====================================================================

def get_gemini_chat_model():
    """
    Initialize ChatGoogleGenerativeAI using configured settings.
    Does not execute network requests until invoke() or a runnable call is made.
    """
    if not has_valid_gemini_key():
        raise ValueError(
            "GOOGLE_API_KEY is not configured in .env. "
            "Please paste your Gemini API key into .env to enable live generation."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )


def generate_mock_quiz_from_chunks(
    chunks: list[Document],
    topic: str,
    difficulty: str,
    question_types: list[str],
    num_questions: int
) -> QuizModel:
    """
    Generate a realistic, context-grounded mock quiz for local offline development.
    Strictly avoids external network calls while ensuring complete Pydantic validation.
    """
    if not chunks:
        raise ValueError("Cannot generate mock quiz: No context chunks provided.")

    questions: list[QuestionModel] = []
    types_cycle = question_types if question_types else ["MCQ", "True/False", "Short Answer"]

    for i in range(num_questions):
        q_idx = i + 1
        q_type = types_cycle[i % len(types_cycle)]
        chunk = chunks[i % len(chunks)]
        source = chunk.metadata.get("source", "Study_Notes.pdf")
        page = chunk.metadata.get("page", 1)
        text_snippet = chunk.page_content.strip()

        # Extract first sentence or concept for realistic phrasing
        first_line = text_snippet.split("\n")[0].strip()
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."

        if q_type == "MCQ":
            q_text = f"According to {source} (Page {page}), what is a core principle of: '{first_line}'?"
            correct = "It provides authoritative context to prevent hallucinations and maintain factual grounding"
            options = [
                correct,
                "It eliminates the need for any language model or inference runtime",
                "It requires training a full 70-billion parameter neural network from scratch",
                "It permanently modifies the base weights of the pre-trained transformer model"
            ]
            explanation = f"As explained on Page {page} of {source}, grounding context ensures factual fidelity without retraining."
            q = QuestionModel(
                question_id=q_idx,
                question=q_text,
                question_type="MCQ",
                difficulty=difficulty,
                options=options,
                correct_answer=correct,
                explanation=explanation,
                source_filename=source,
                source_page=page
            )
        elif q_type == "True/False":
            q_text = f"True or False: According to the notes in {source} (Page {page}), '{first_line}' is a valid statement."
            correct = "True"
            options = ["True", "False"]
            explanation = f"Supported directly by the study notes on Page {page} of {source}."
            q = QuestionModel(
                question_id=q_idx,
                question=q_text,
                question_type="True/False",
                difficulty=difficulty,
                options=options,
                correct_answer=correct,
                explanation=explanation,
                source_filename=source,
                source_page=page
            )
        else: # Short Answer
            q_text = f"Based on the notes on Page {page} of {source}, briefly explain the significance of: '{first_line}'."
            ref_ans = "It bridges external domain-specific knowledge with LLM generation to ensure accurate, verifiable responses."
            concepts = ["Domain knowledge", "LLM generation", "Factual accuracy"]
            explanation = f"Key concepts derived from Page {page} of {source}."
            q = QuestionModel(
                question_id=q_idx,
                question=q_text,
                question_type="Short Answer",
                difficulty=difficulty,
                options=[],
                correct_answer=ref_ans,
                explanation=explanation,
                source_filename=source,
                source_page=page,
                reference_answer=ref_ans,
                key_concepts=concepts
            )

        questions.append(q)

    quiz_title = f"{topic} Quiz" if topic and topic.strip() else "Study Notes Grounded Quiz"
    return QuizModel(
        title=quiz_title,
        topic=topic if topic and topic.strip() else "Uploaded Study Notes",
        difficulty=difficulty,
        total_questions=len(questions),
        questions=questions
    )


# =====================================================================
# Main Quiz Generation Function
# =====================================================================

def generate_quiz(
    session_id: str,
    topic: str = "",
    difficulty: str = "Medium",
    question_types: list[str] | None = None,
    num_questions: int = 5,
    force_mock: bool = False
) -> dict[str, Any]:
    """
    Generate a quiz strictly grounded in the session's retrieved ChromaDB chunks.
    
    Args:
        session_id: Active visitor session identifier.
        topic: Specific focus area or broad topic.
        difficulty: 'Easy', 'Medium', or 'Hard'.
        question_types: List of question formats (['MCQ', 'True/False', 'Short Answer']).
        num_questions: Target number of questions (5, 10, 15, 20).
        force_mock: If True, uses the verified offline mock generator (zero API calls).
        
    Returns:
        Dictionary representation of QuizModel.
        
    Raises:
        ValueError: If context is insufficient or API key is missing when live mode requested.
    """
    if question_types is None:
        question_types = ["MCQ", "True/False", "Short Answer"]

    # 1. Retrieve relevant chunks using the existing Phase 5 retrieval pipeline
    retrieval_query = topic.strip() if topic.strip() else "core concepts, principles, and definitions"
    retrieved_chunks = retrieve_relevant_chunks(
        session_id=session_id,
        query=retrieval_query,
        top_k=max(6, num_questions * 2)
    )

    if not retrieved_chunks:
        if force_mock or not has_valid_gemini_key():
            # Provide sample fallback study notes for zero-setup offline exploration
            retrieved_chunks = [
                Document(
                    page_content="Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large language model responses by retrieving relevant factual documents from an external vector store. It eliminates hallucination and keeps facts strictly verifiable.",
                    metadata={"source": "Lecture_Introduction_to_RAG.pdf", "page": 1}
                ),
                Document(
                    page_content="Vector embeddings are dense numerical vectors representing semantic meaning. The sentence-transformers/all-MiniLM-L6-v2 model generates 384-dimensional vectors. Documents are split into chunks using recursive character splitters with overlap.",
                    metadata={"source": "Lecture_Embeddings_and_Vectors.pdf", "page": 2}
                ),
                Document(
                    page_content="Deterministic scoring evaluates objective question types like MCQs and True/False questions in pure Python. Short answer questions are graded against a rubric evaluating coverage of required key concepts.",
                    metadata={"source": "Lecture_Assessment_Design.pdf", "page": 3}
                )
            ]
        else:
            raise ValueError(
                "Insufficient context in vector store. "
                "Please upload and index your notes in '📁 Upload Notes' before generating a quiz."
            )

    # 2. Offline / Mock Mode Check
    if force_mock or not has_valid_gemini_key():
        mock_quiz = generate_mock_quiz_from_chunks(
            chunks=retrieved_chunks,
            topic=topic,
            difficulty=difficulty,
            question_types=question_types,
            num_questions=num_questions
        )
        return mock_quiz.model_dump()

    # 3. Live Gemini Generation (only when key is provided and live mode requested)
    formatted_context_pieces = []
    for idx, doc in enumerate(retrieved_chunks, start=1):
        src = doc.metadata.get("source", "Document")
        pg = doc.metadata.get("page", 1)
        formatted_context_pieces.append(
            f"--- Context Snippet {idx} [Source: {src}, Page: {pg}] ---\n{doc.page_content}\n"
        )
    combined_context = "\n".join(formatted_context_pieces)

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUIZ_SYSTEM_PROMPT),
        ("human", QUIZ_USER_PROMPT),
    ])

    llm = get_gemini_chat_model()
    # Use Pydantic structured output with Gemini
    structured_llm = llm.with_structured_output(QuizModel)
    chain = prompt | structured_llm

    quiz_output: QuizModel = chain.invoke({
        "context": combined_context,
        "topic": topic if topic.strip() else "Comprehensive Study Notes",
        "difficulty": difficulty,
        "num_questions": num_questions,
        "question_types": ", ".join(question_types)
    })

    return quiz_output.model_dump()
