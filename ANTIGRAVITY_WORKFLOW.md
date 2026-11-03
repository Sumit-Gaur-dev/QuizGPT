# Quiz RAG Application --- Antigravity Development Blueprint

> **Purpose:** Build a small, understandable Quiz RAG application that
> demonstrates the complete Retrieval-Augmented Generation (RAG)
> workflow. The priority is learning and demonstrating RAG---not
> building a production SaaS platform.

## 1. Project goal

A user uploads their study notes as PDFs, the application extracts and
indexes the content, and the user generates quizzes grounded in those
notes. The user attempts a quiz and receives a score and feedback. The
questions and explanations must be supported by the uploaded material.

### What success looks like

A user can:

1.  Open the Streamlit application.
2.  Upload one or more text-based PDF notes.
3.  See the uploaded files and ingestion status.
4.  Generate a quiz from the uploaded notes.
5.  Answer the questions.
6.  Submit the quiz and see their score and feedback.
7.  Review the questions, correct answers, and supporting context.
8.  Delete their notes and quiz data when finished.

The application must make the RAG steps visible enough to demonstrate
them: **document → extracted text → chunks → embeddings → vector store →
retrieval → grounded generation**.

## 2. Scope and non-goals

### In scope

-   Streamlit user interface.
-   Python-based PDF extraction and RAG pipeline.
-   Chunking with configurable chunk size and overlap.
-   Embedding generation.
-   ChromaDB vector storage and similarity retrieval.
-   Gemini API for quiz generation and short-answer evaluation.
-   MCQ, True/False, and short-answer question types.
-   Quiz submission, scoring, and feedback.
-   Session-scoped temporary data.
-   Clear errors, loading states, and basic validation.
-   Local testing before deployment.

### Explicitly out of scope for this version

-   User accounts, login, and authentication.
-   React, Node.js, Express, or a separate backend service.
-   A separate SQL database.
-   Payments, subscriptions, or admin dashboards.
-   Multi-tenant production infrastructure.
-   Permanent quiz history across sessions or server restarts.
-   Complex agent frameworks, LangChain, or MCP.
-   OCR for scanned PDFs in the first version.
-   Match-the-columns questions unless the core question types are
    working reliably.

Do not add out-of-scope features unless the user explicitly approves
them.

## 3. Proposed technology stack

  -----------------------------------------------------------------------
  Layer                   Technology              Responsibility
  ----------------------- ----------------------- -----------------------
  UI and app runtime      Streamlit               Pages, controls,
                                                  session state,
                                                  rendering

  Language                Python                  Application and RAG
                                                  logic

  PDF extraction          PyMuPDF (`fitz`)        Extract text from PDFs

  Chunking                Python                  Split extracted text
                                                  into overlapping chunks

  Embeddings              Gemini embedding API    Convert chunks and
                                                  queries into vectors

  Vector database         ChromaDB Python library Store vectors, text,
                                                  and metadata; retrieve
                                                  relevant chunks

  LLM                     Gemini API via          Generate quizzes and
                          `google-genai`          evaluate short answers

  Configuration           `python-dotenv`         API key and settings
                          locally; Streamlit      
                          secrets when hosted     

  Local storage           ChromaDB                Persist local
                          `PersistentClient`      development data
                          during local            
                          development             

  Hosted storage          Session-scoped          Temporary, isolated
                          `EphemeralClient`       hosted session data
  -----------------------------------------------------------------------

**Model/configuration rule:** Do not scatter model names through the
code. Put model identifiers and chunk/retrieval settings in one
configuration module. Verify currently supported Gemini model
identifiers and SDK usage when implementing; do not assume an old model
name is still available.

**Embedding consistency rule:** Use the same embedding model and
compatible embedding configuration for both document chunks and user
queries. Never compare vectors created by different embedding models.

## 4. Architecture

``` text
                    Streamlit Application
                            |
                 User uploads PDF notes
                            |
                     PDF extraction
                            |
                       Chunking
                            |
                    Embedding model
                            |
                   ChromaDB collection
                            |
              User requests quiz generation
                            |
                  Embed retrieval query
                            |
                  Similarity retrieval
                            |
              Retrieved chunks + instructions
                            |
                     Gemini LLM
                            |
                  Structured quiz output
                            |
                    Streamlit quiz UI
                            |
                 User submits their answers
                            |
                 Scoring and evaluation
                            |
                 Results + grounded feedback
```

### Important separation

-   `frontend/` contains Streamlit rendering and user interaction.
-   `rag/` contains document processing, retrieval, generation, and
    evaluation logic.
-   `config/` contains settings and session/database initialization.
-   `app.py` starts the app and coordinates the UI.

Keep the RAG functions independent of Streamlit wherever practical. A
function in `rag/` should accept normal Python inputs and return normal
Python outputs.

## 5. Suggested project structure

``` text
quiz-rag/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── frontend/
│   ├── dashboard.py
│   ├── upload.py
│   ├── generate_quiz.py
│   ├── quiz_attempt.py
│   └── quiz_results.py
│
├── rag/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── generation.py
│   └── evaluation.py
│
├── config/
│   ├── settings.py
│   ├── chroma_config.py
│   └── session.py
│
└── storage/
    └── chroma_db/       # Created locally at runtime; do not commit database files
```

Antigravity may suggest a simpler structure if it has a clear reason,
but it must preserve the separation between UI, RAG logic,
configuration, and runtime storage.

## 6. Data and session lifecycle

This application does not have user accounts. A Streamlit browser
session is the temporary boundary for a visitor's data.

### Local development

-   Use a local `PersistentClient` so the developer can restart the app
    without losing the local test collection.
-   Keep the local database directory out of Git.
-   Provide a clear **Delete my data** action that deletes the current
    session's documents and quiz state.
-   Use a unique collection per Streamlit session, or another design
    that prevents one session from querying another session's chunks.

### Hosted demo

-   Use an in-memory, session-scoped ChromaDB client/collection for the
    simplest temporary demo.
-   Store the session's client/collection identifier and quiz state in
    `st.session_state`.
-   Do not use one shared global collection for all visitors.
-   Do not promise permanent storage. Temporary hosted data can
    disappear when a session ends or the app restarts.
-   Do not store uploaded PDFs or user content in a shared directory
    that another visitor can access.

### Session state

Keep only the state required by the current visitor, such as:

-   session identifier
-   ChromaDB client/collection reference or safe collection identifier
-   uploaded file names
-   ingestion status
-   current quiz
-   current answers
-   latest result

Do not assume `st.session_state` is permanent storage. It is temporary
application session state.

### Deletion

When the user clicks **Delete my data**:

1.  Delete the current session's ChromaDB collection or its
    session-specific records.
2.  Clear uploaded-file references and quiz state from
    `st.session_state`.
3.  Clear any temporary files created for that session.
4.  Show a confirmation message.

Ensure deletion is scoped to the current session only.

## 7. Functional requirements

### A. Upload notes

-   Accept PDF files only in the first version.
-   Allow multiple PDFs if feasible.
-   Validate file type and reject empty or unreadable PDFs.
-   Set a configurable upload-size limit.
-   Display file names and ingestion progress.
-   Extract text from each PDF.
-   Detect scanned/image-only PDFs and explain that OCR is not supported
    in V1.
-   Preserve source file name and page number as chunk metadata where
    available.

### B. Text chunking

-   Split extracted text into manageable chunks.
-   Use configurable chunk size and overlap.
-   Preserve source metadata on every chunk.
-   Avoid empty chunks.
-   Avoid silently dropping extracted content.
-   Show a small preview of extracted text and chunks for
    learning/debugging.

Start with a reasonable configurable baseline, for example: - chunk
size: approximately 800--1,000 tokens (or a clearly documented
character-based equivalent) - overlap: approximately 100--150 tokens

Do not mix up characters and tokens. If the implementation uses
character counts, label and document the settings as characters.

### C. Embeddings and ChromaDB

-   Generate embeddings for each chunk.
-   Store chunk text, embedding, and metadata in ChromaDB.
-   Use a stable unique ID for every chunk.
-   Store source filename and page number in metadata.
-   Embed the retrieval query using the same embedding model.
-   Retrieve the top-K most relevant chunks.
-   Display retrieved snippets and source metadata in a debug/learning
    panel.
-   Handle empty collections and embedding/API errors gracefully.

Do not create a second database just to store chunk text: ChromaDB can
store document text and metadata alongside embeddings.

### D. Quiz generation

The user should be able to select:

-   Question type: MCQ, True/False, or Short Answer.
-   Number of questions.
-   Difficulty: Easy, Medium, or Hard.

For every generation request:

1.  Create a retrieval query from the selected topic/instructions, or a
    broad query if no topic is supplied.
2.  Retrieve relevant chunks from the current session's collection.
3.  Send only the retrieved context, question settings, and strict
    output instructions to Gemini.
4.  Request structured JSON output.
5.  Validate the returned structure before displaying the quiz.
6.  If the response is invalid, handle it safely; do not crash the app
    or display malformed questions.

Each question should contain: - stable question ID - question text -
question type - options for MCQ - correct answer - short explanation -
supporting source filename/page or chunk references

**Grounding rules for the generation prompt:**

-   Use only the supplied retrieved context.
-   Do not use outside knowledge to invent facts.
-   If the context does not contain enough information, return fewer
    questions or clearly report insufficient context.
-   Do not create duplicate or near-duplicate questions.
-   Ensure the correct answer is actually supported by the context.
-   For MCQs, make one option unambiguously correct and plausible
    distractors.
-   Do not reveal the correct answer in the question text or options.
-   Include evidence/source references for each question.

### E. Quiz attempt

-   Render each question according to its type.
-   Keep user answers in session state.
-   Allow the user to change answers before submission.
-   Prevent accidental loss of answers during Streamlit reruns.
-   Show a clear Submit Quiz action.
-   Do not show correct answers before submission.

### F. Scoring and feedback

**MCQ and True/False** - Score deterministically in Python by comparing
normalized answers. - Do not call Gemini just to mark objective answers.

**Short Answer** - Evaluate using Gemini with: - the question - the
user's answer - expected answer or rubric - supporting retrieved
context - Ask for structured output containing correctness/score,
explanation, and missing key points. - The evaluator must not award
credit for claims contradicted by the source context. - Treat LLM
evaluation as approximate and explain that it is AI-assisted.

**Results screen** - Display total score and correct/incorrect counts. -
Show each question, user's answer, expected answer, explanation, and
source evidence. - Give concise feedback on weak areas. - Do not claim
the feedback is a formal academic assessment.

## 8. Development workflow --- build and verify one phase at a time

**Critical instruction to Antigravity:** Do not implement the entire
application in one large code-generation step. Work phase by phase. At
the end of each phase, explain what changed, list files
created/modified, provide exact commands to run, and wait for the
developer to test and approve before starting the next phase.

### Phase 0 --- Confirm plan and scaffold

Build: - project folder structure - virtual environment instructions -
`requirements.txt` - `.env.example` - `.gitignore` - minimal `app.py`

Developer checks: - dependencies install successfully - Streamlit app
starts locally - no API key is committed - folder structure is
understandable

**Stop and wait for approval.**

### Phase 1 --- Streamlit UI shell

Build: - page configuration and title - simple dashboard - PDF upload
area - quiz settings controls - empty quiz/results states - clear
navigation or sections

Use mock/sample data only. Do not implement RAG yet.

Developer checks: - app launches - layout is usable - controls respond -
no Streamlit errors - UI remains understandable at normal desktop size

**Stop and wait for approval.**

### Phase 2 --- PDF extraction

Build: - PDF validation - text extraction module - source/page
metadata - readable error messages - extracted-text preview

Developer checks: - test with a normal text PDF - test with multiple
PDFs - test with an empty or invalid file - test with an
image-only/scanned PDF - verify page/source metadata

**Stop and wait for approval.**

### Phase 3 --- Chunking

Build: - chunking function - configurable chunk size and overlap -
metadata propagation - chunk preview

Developer checks: - inspect several chunks manually - verify overlap
works - verify no empty chunks - verify source filename/page metadata is
retained - verify short documents are handled correctly

**Stop and wait for approval.**

### Phase 4 --- Embeddings and ChromaDB

Build: - embedding client - ChromaDB initialization - add/upsert chunk
records - query embedding - top-K retrieval - session isolation and
deletion

Developer checks: - ingest a small PDF - confirm collection contains
expected records - run a retrieval query - inspect returned text and
metadata - confirm a relevant query retrieves relevant chunks - confirm
a second session cannot retrieve the first session's data - confirm
delete action removes only the current session's data

**Stop and wait for approval.**

### Phase 5 --- Gemini quiz generation

Build: - Gemini client - prompt template - structured response schema -
response parsing and validation - quiz generation UI integration -
source evidence per question

Developer checks: - generate MCQ questions from a known PDF - generate
True/False questions - generate short-answer questions - verify answers
against the PDF - try an unrelated topic and check that the system does
not invent unsupported facts - try malformed/failed API response
handling - inspect retrieved context used for generation

**Stop and wait for approval.**

### Phase 6 --- Quiz attempts and evaluation

Build: - question rendering - answer state management - objective
scoring - short-answer evaluation - results and feedback screen

Developer checks: - answer all questions correctly - answer all
questions incorrectly - leave some questions unanswered - change an
answer before submission - verify objective score calculations
manually - verify short-answer feedback is tied to the supplied
context - refresh/rerun during an attempt and inspect state behavior

**Stop and wait for approval.**

### Phase 7 --- Full local integration test

Do not add new features in this phase. Fix issues found during testing.

Run the complete journey:

1.  Start the app locally.
2.  Upload a known PDF.
3.  Inspect extracted text.
4.  Inspect chunks and metadata.
5.  Ingest chunks into ChromaDB.
6.  Run a retrieval query and inspect its results.
7.  Generate a quiz.
8.  Verify each question against the original notes.
9.  Attempt the quiz.
10. Submit and inspect the score and feedback.
11. Delete the session's data.
12. Verify that the deleted data is no longer retrievable.

Also check: - missing API key - API quota/rate-limit failure - network
failure - unsupported PDF - empty PDF text - duplicate upload - no
retrieved results - invalid LLM output - session isolation - large PDF
handling within configured limits

**Deployment is blocked until the developer explicitly approves the
local application.**

### Phase 8 --- Deployment preparation and deployment

Only after local testing is approved:

-   pin compatible dependency versions
-   verify the app runs from a clean environment
-   add Gemini API key using hosting secrets, never Git
-   ensure `.env`, database files, uploads, and secrets are excluded
    from Git
-   choose the hosted session-storage approach
-   deploy to Streamlit Community Cloud
-   verify the deployed app with a small non-sensitive sample PDF
-   verify session isolation and deletion on the hosted app
-   document hosted data-loss behavior clearly

Do not promise permanent hosted storage when using ephemeral storage.

## 9. Local run instructions

Antigravity should tailor commands to the Python environment available
on the developer's machine. Example using `uv`:

``` bash
uv init
uv venv
```

Activate the environment using the command appropriate for the operating
system, then install dependencies from `requirements.txt`.

Example:

``` bash
uv pip install -r requirements.txt
streamlit run app.py
```

Create a local `.env` file based on `.env.example`:

``` env
GEMINI_API_KEY=your_key_here
```

Never put a real API key in source code, README examples, screenshots,
or Git commits.

## 10. Configuration expectations

Keep settings centralized. Examples of configurable values:

-   Gemini generation model
-   Gemini embedding model
-   chunk size
-   chunk overlap
-   retrieval top-K
-   maximum PDF upload size
-   maximum number of PDFs per session
-   maximum total pages per session
-   maximum questions per quiz

Start with conservative limits. Do not claim a universal maximum
PDF/page count based only on hosting disk capacity. The actual limit
depends on RAM, PDF size, extraction time, embedding API quotas, request
limits, and the hosting platform's resource allocation.

## 11. Reliability, privacy, and cost guardrails

-   Never expose the Gemini API key to the browser.
-   Never commit `.env`, uploaded PDFs, or ChromaDB files.
-   Keep collections isolated by Streamlit session.
-   Do not use a single global collection shared by visitors.
-   Validate all user-uploaded files.
-   Limit PDF size, page count, and quiz size to prevent resource
    exhaustion.
-   Handle Gemini rate limits and quota errors with a useful message.
-   Avoid repeated embedding calls for unchanged chunks during one
    ingestion.
-   Avoid embedding the same text twice when a duplicate upload is
    detected.
-   Do not send the entire PDF to Gemini when retrieval can provide
    relevant context.
-   Do not treat LLM-generated answers as automatically correct; verify
    against source context.
-   Keep debug output useful but do not print API keys or sensitive
    document contents to server logs.

## 12. Definition of done

The V1 application is complete when:

-   [ ] A user can upload a supported PDF.
-   [ ] Text extraction works and reports unsupported scanned PDFs.
-   [ ] Text is split into inspectable chunks.
-   [ ] Embeddings are generated consistently.
-   [ ] ChromaDB stores chunks and metadata.
-   [ ] Retrieval returns relevant source chunks.
-   [ ] Gemini generates structured, source-grounded quizzes.
-   [ ] MCQ and True/False scoring is deterministic.
-   [ ] Short-answer feedback is context-grounded and clearly
    AI-assisted.
-   [ ] Results show score, explanations, and source evidence.
-   [ ] Session data is isolated between visitors.
-   [ ] A user can delete their own temporary data.
-   [ ] The full workflow passes local manual testing.
-   [ ] Deployment happens only after explicit developer approval.

## 13. Master instruction for Antigravity

Use this as the initial prompt when opening the project in Antigravity:

------------------------------------------------------------------------

**You are my pair programmer for a beginner-friendly Python RAG project
called Quiz RAG.**

Read this entire `ANTIGRAVITY_WORKFLOW.md` before writing code. Treat it
as the project specification and implementation plan.

### How you must work

1.  First inspect the existing workspace. Do not overwrite existing
    files without explaining why.
2.  Summarize the architecture and development phases in your own words.
3.  Identify any missing decisions or technical incompatibilities before
    implementation.
4.  Start with Phase 0 only.
5.  Implement only the current approved phase.
6.  Keep the code modular, readable, and beginner-friendly. I am
    learning Python and RAG, so avoid unnecessary abstractions and
    clever patterns.
7.  Explain the purpose of each important file and function.
8.  Do not add technologies or features outside the agreed scope without
    asking me.
9.  Do not silently skip errors, tests, or manual verification steps.
10. At the end of each phase, provide:

-   what you implemented
-   files created or modified
-   important implementation decisions
-   exact commands I should run
-   a short manual test checklist
-   known limitations or issues

11. Then STOP and wait for my explicit approval before proceeding to the
    next phase.
12. If my manual test fails, help me diagnose and fix the current phase
    before moving forward.
13. Do not deploy, publish, push secrets, or modify deployment settings
    until I explicitly approve the deployment phase.

### Technical constraints

-   Use Streamlit and Python for the application.
-   Use ChromaDB as a Python library; do not require ChromaDB Cloud.
-   Use Gemini for quiz generation and short-answer evaluation.
-   Keep document processing and RAG logic separate from Streamlit UI
    code.
-   Use consistent embeddings for document chunks and retrieval queries.
-   Keep visitor data session-isolated. Never expose one visitor's
    uploaded notes to another visitor.
-   Use local persistent ChromaDB storage during local development and
    an appropriate temporary session-scoped approach for the hosted
    demo.
-   Keep secrets out of source control.
-   Do not introduce authentication, SQL databases, React, Node.js,
    LangChain, or MCP in V1.

### Immediate task

Begin with **Phase 0 --- Confirm plan and scaffold**. Do not implement
Phase 1 or any RAG functionality yet. First report your understanding
and any important concerns, then create the minimal scaffold and give me
the commands to run it. Wait for my test results and approval.

------------------------------------------------------------------------

## 14. Developer operating rule

**The developer is the release gate.** Antigravity can implement and
explain, but it must not move to the next phase until the developer has
run the application, checked the current phase, and explicitly approved
continuing.

The goal is not to generate the maximum amount of code quickly. The goal
is to build a working RAG application while understanding every major
part of the pipeline.
