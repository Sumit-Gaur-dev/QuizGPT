# Quiz RAG - AI-Powered Study Assistant

A beginner-friendly Retrieval-Augmented Generation (RAG) application that lets users upload study notes as PDFs, indexes them into a local vector database, and generates quizzes strictly grounded in those notes.

Built with **Streamlit**, **PyMuPDF**, **ChromaDB**, and the **Google Gemini API**.

---

## Architecture Overview

```text
                  Streamlit Application (frontend/)
                                 |
                      User uploads PDF notes
                                 |
                 PDF extraction (rag/ingestion.py)
                                 |
                    Chunking (rag/chunking.py)
                                 |
                 Embedding model (rag/embeddings.py)
                                 |
                ChromaDB collection (config/chroma_config.py)
                                 |
                   User requests quiz generation
                                 |
                 Similarity retrieval (rag/retrieval.py)
                                 |
              Gemini LLM Generation (rag/generation.py)
                                 |
                 Streamlit quiz UI (frontend/quiz_attempt.py)
                                 |
             Scoring & Evaluation (rag/evaluation.py)
                                 |
             Results & Grounded Feedback (frontend/quiz_results.py)
```

### Folder Structure

```text
QuizGPT/
├── app.py                  # Streamlit entry point and navigation coordinator
├── requirements.txt        # Pinned project dependencies
├── .env.example            # Sample environment variables
├── .gitignore              # Git ignore rules (secrets, venv, chromadb storage)
├── README.md               # Project documentation
│
├── frontend/               # Streamlit UI modules
├── rag/                    # Pure Python RAG pipeline modules
├── config/                 # Centralized configuration & session handlers
└── storage/                # Local database storage (excluded from git)
```

---

## Quickstart (Phase 0)

### 1. Create and Activate Virtual Environment

Using `uv` (recommended):
```powershell
uv venv
.\.venv\Scripts\activate
```

Or using standard Python `venv`:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies

Using `uv`:
```powershell
uv pip install -r requirements.txt
```

Or using standard `pip`:
```powershell
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Copy the example file:
```powershell
Copy-Item .env.example .env
```
*(You won't need a real Gemini API key until Phase 4/5, but having `.env` ready prevents missing configuration errors later).*

### 4. Run the Application

```powershell
streamlit run app.py
```
Visit `http://localhost:8501` to view the Phase 0 scaffold verification screen.
