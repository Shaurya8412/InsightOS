# InsightOS: Intelligent Grounded Knowledge RAG System

InsightOS is a local, lightweight Research and Knowledge synthesis engine. It implements a robust, text-first Retrieval-Augmented Generation (RAG) pipeline designed to ingest, process, store, and query complex unstructured documents (PDFs, Markdown, and text files) with deterministic, traceable citations.

## The Problem InsightOS Solves
Knowledge workers, analysts, and developers waste hours cross-referencing information across multiple dense documents. Traditional out-of-the-box LLMs suffer from two critical limitations:
1. **The "Black Box" Retrieval Gap:** There is no transparency into *what* specific chunks of text the LLM read before responding.
2. **Untraceable Citations:** LLMs routinely hallucinate reference links or offer generalized, unverifiable sourcing.

InsightOS makes the retrieval pipeline explicit, enforces strict grounding (preventing responses when source documents lack the answer), and provides deterministic citation links mapping retrieved text directly back to its source document and page number.

---

## High-Level Architecture
InsightOS is built with a highly decoupled, modular architecture without relying on heavy wrappers like LangChain or LlamaIndex:

```
[User Ingestion]  --->  [Parsing via PyMuPDF] ---> [Structural Chunking] 
                                                               |
                                                               v
[Qdrant Cloud Vector DB] <--- [Gemini Embeddings] <--- [Data Ingestion Service]
          |
          v
[User Natural Query] ---> [Semantic Top-K Search] ---> [Grounded Prompt Construction] ---> [Gemini LLM Synthesis] ---> [Grounded Answer + Metadata Citations]
```

1. **Ingestion & Parsing:** Uploaded documents are parsed using PyMuPDF and split into deterministic segments with exact metadata tracking (`document_id`, `page_number`, `source_location`).
2. **Database Persistence:** Metadata is logged to a local SQLite tracking database (using SQLAlchemy). Text chunks are embedded and indexed into Qdrant Cloud.
3. **Retrieval & Orchestration:** Questions trigger semantic similarity searches in Qdrant Cloud. A custom Python orchestrator formats a strict context-bounded prompt.
4. **Grounded Generation:** Gemini processes the prompt, returning a structured response that references verified sources.

---

## Repository Structure
The repository is structured to maintain a clean separation of concerns:

```
InsightOS/
│
├── src/
│   ├── api/             # FastAPI REST endpoints and routes
│   ├── core/            # App configurations, database connections, and custom exceptions
│   ├── models/          # SQLAlchemy DB models and Pydantic schemas
│   ├── services/        # Business logic services (embedding, ingestion, RAG, vector store)
│   │   ├── embeddings/  # Gemini Embedding API wrapper
│   │   ├── ingestion/   # Document parsing and text chunking logic
│   │   ├── rag/         # Orchestrator, citation resolver, and LLM generator
│   │   └── vector_store/# Qdrant client connection and operation handlers
│   └── main.py          # FastAPI application entry point
│
├── tests/
│   ├── integration/     # E2E pipeline integration tests (running on live APIs)
│   └── unit/            # Isolated mock unit tests for api, service layers, and store
│
├── .gitignore           # Excludes local configuration, databases, test caches, and env files
├── README.md            # Product specification and developer guide
├── pyproject.toml       # Dependency declaration and python configuration
└── uv.lock              # Lock file for deterministic python environments
```

---

## Main Technologies Used
*   **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) with Pydantic for validation.
*   **Embedding & LLM APIs:** [Google GenAI SDK](https://github.com/google/generative-ai-python) (`text-embedding-004` and `gemini-3.6-flash`).
*   **Vector Database:** [Qdrant Cloud](https://qdrant.tech/) via the official python client wrapper.
*   **Document Processor:** [PyMuPDF](https://pymupdf.readthedocs.io/) for fast, robust text extraction.
*   **Metadata DB:** [SQLAlchemy](https://www.sqlalchemy.org/) + [SQLite](https://www.sqlite.org/).
*   **Frontend UI:** [Streamlit](https://streamlit.io/) for a simple upload and research interface.
*   **Environment & Python Runner:** [uv](https://github.com/astral-sh/uv) (from Astral) for blazingly fast dependency management.

---

## Installation & Local Configuration

### 1. Prerequisites
Ensure you have Python 3.11+ installed. It is recommended to use `uv` for setup:
```bash
pip install uv
```

### 2. Clone and Initialize Virtual Environment
Initialize the environment and sync dependencies:
```bash
uv venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate    # On macOS/Linux
uv pip install -e .
```

### 3. Environment Variables (.env)
Create a `.env` file in the project root:
```env
# Database Settings
SQLITE_DB_URL="sqlite:///./insightos.db"

# API Keys (Never commit these to git!)
GEMINI_API_KEY="your-gemini-api-key"
QDRANT_API_KEY="your-qdrant-cluster-api-key"
QDRANT_URL="https://your-qdrant-cluster-url.aws.cloud.qdrant.io"

# Default Model Overrides (Optional)
LLM_MODEL="gemini-3.6-flash"
EMBEDDING_MODEL="text-embedding-004"
```

---

## Running the Application

### 1. Initialize SQLite Tables
Run the database creation step to initialize SQLite schemas:
```bash
.venv\Scripts\python.exe -c "from src.core.database import init_db; from src.models.db_models import Document; init_db()"
```

### 2. Start FastAPI Backend
```bash
.venv\Scripts\python.exe -m uvicorn src.main:app --port 8000 --reload
```
You can verify the backend is up and running by opening `http://127.0.0.1:8000/health`.

### 3. Start Streamlit Frontend
In a new terminal window:
```bash
.venv\Scripts\python.exe -m streamlit run src/frontend/app.py
```
Open `http://127.0.0.1:8501` in your browser to interact with the UI.

---

## Testing & Verification
We maintain a comprehensive suite of unit (using mocks) and integration (running on live endpoints) tests.

Run the entire test suite:
```bash
.venv\Scripts\python.exe -m pytest
```

Run only unit tests (safe to run without internet or API credentials):
```bash
.venv\Scripts\python.exe -m pytest tests/unit/
```

Run live integration tests (requires valid `.env` credentials):
```bash
.venv\Scripts\python.exe -m pytest tests/integration/
```

---

## Important Development Notes
*   **Vector Normalization:** In Qdrant, Cosine similarity vectors are L2-normalized upon ingestion. For custom unit/integration tests matching vectors, utilize unit vectors (e.g. `[1.0] + [0.0] * 127`) to avoid normalization discrepancies.
*   **Payload Indexing:** Qdrant Cloud requires a payload index for filtering operations (including delete with filter) on custom attributes like `document_id`. The application automatically indexes this field upon collection initialization.
*   **No-Context Enforcement:** Prompts bound the LLM strictly to the provided document context. If a user query cannot be resolved via retrieved chunks, the orchestrator returns a standard grounding fallback rather than guessing or hallucinating answers.
