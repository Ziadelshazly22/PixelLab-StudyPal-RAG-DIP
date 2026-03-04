# Smart Learning Assistant

> **RAG-Powered AI Tutor for Digital Image Processing**

A production-ready microservice that grounds every answer in the *Gonzalez & Woods* textbook and verified library documentation (OpenCV, scikit-image). It uses a **dual-LLM strategy** — Gemini 2.0 Flash for fast, grounded responses and DeepSeek-R1 for deep mathematical reasoning — delivered through a clean REST API and an interactive Gradio chat interface.

Built with **LangChain · LangServe · FastAPI · ChromaDB · Gradio**.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Module Overview](#module-overview)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Architecture

```plaintext
User (browser / API client)
        │
        ▼
┌───────────────────────────────────────────┐
│          FastAPI  (main.py)               │
│  ┌──────────────┐  ┌─────────────────┐   │
│  │  REST routes │  │  LangServe /rag │   │
│  │  /api/*      │  │  /summarize     │   │
│  └──────────────┘  └─────────────────┘   │
│           │                │             │
│           ▼                ▼             │
│     app/api/        app/chains/          │
│     router.py       rag_chain.py         │
│                          │               │
│                    app/retrieval/        │
│                    retriever.py          │
│                          │               │
│                    ChromaDB (local)      │
│                    ← app/ingestion/ ─►   │
│                      pipeline.py         │
└───────────────────────────────────────────┘
        │
        ▼
   Gradio UI  (/ui)
```

**Ingestion pipeline:** `data/raw/*.pdf` → PyMuPDF loader → `RecursiveCharacterTextSplitter` → embeddings *(Google `text-embedding-004` primary / `all-MiniLM-L6-v2` fallback)* → ChromaDB

**Query pipeline:** Question → MMR retrieval → LCEL chain (prompt + LLM) → cited answer

---

## Project Structure

```text
smart-learning-assistant/
├── app/
│   ├── api/                   # FastAPI routers
│   │   ├── __init__.py
│   │   └── router.py
│   ├── chains/                # LangChain RAG chains
│   │   ├── __init__.py
│   │   └── rag_chain.py
│   ├── ingestion/             # PDF parsing, chunking, embedding
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── retrieval/             # Retriever logic
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── summarization/         # Chapter summarization chains
│   │   ├── __init__.py
│   │   └── summarizer.py
│   ├── evaluation/            # Metrics, RAGAS scripts
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── ui/                    # Gradio interface
│       ├── __init__.py
│       └── interface.py
├─── data/
│    ├── chroma_db/             # Persistent Chroma vector store
│    └── raw/
│        │
│        ├── 1_textbooks/
│        │   │
│        │   └── Digital_Image_Processing_Gonzalez_Woods_4th_Ed.pdf
│        │
│        ├── 2_core_vision/
│        │   │
│        │   ├── opencv2ref.pdf
│        │   │ 
│        │   ├── numpy-user.pdf
│        │   │
│        │   └── scipy-ref.pdf
│        │
│        └── 3_python_utilities/
│            │  
│            ├── Matplotlib.pdf
│            │     
│            └── pillow.pdf    
│     
├── notebooks/                 # Prototyping notebooks
├── tests/                     # Unit tests
│   └── __init__.py
├── .env.example               # Environment variable template
├── .gitignore
├── main.py                    # FastAPI + LangServe entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python **3.10+** (project tested on 3.12)
- A [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
- *(Optional)* A DeepSeek API key

### 1. Clone & create environment

```bash
git clone https://github.com/Ziadelshazly22/PixelLab-StudyPal-RAG-DIP.git
cd PixelLab-StudyPal-RAG-DIP/smart-learning-assistant

# Windows
py -3 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

### 4. Ingest your documents

Drop PDF files (e.g. *Gonzalez & Woods – Digital Image Processing*) into `data/raw/`, then run:

```bash
python -m app.ingestion.pipeline
```

### 5. Start the server

```bash
python main.py
# or with auto-reload during development:
uvicorn main:app --reload
```

| Endpoint | URL |
| --- | --- |
| API root | `http://localhost:8000/` |
| Interactive docs (Swagger) | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| Gradio chat UI | `http://localhost:8000/ui` |

---

## Environment Variables

Copy `.env.example` to `.env` and populate each value:

| Variable | Required | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | ✅ | Gemini API key — [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `EMBEDDING_MODEL` | ✅ | Embedding model name (default: `models/text-embedding-004`) |
| `LLM_MODEL` | ✅ | Primary LLM model name (default: `gemini-2.0-flash`) |
| `CHROMA_PERSIST_DIR` | ✅ | Path to ChromaDB storage (default: `./data/chroma_db`) |
| `OLLAMA_BASE_URL` | ⬜ | Ollama server URL for DeepSeek-R1 fallback (default: `http://localhost:11434`) |
| `DEEPSEEK_MODEL` | ⬜ | DeepSeek model name served by Ollama (default: `deepseek-r1`) |

---

## Module Overview

| Module | File | Responsibility |
| --- | --- | --- |
| `app.api` | `router.py` | Auxiliary REST endpoints (`/api/health`, `/api/info`) |
| `app.chains` | `rag_chain.py` | LCEL RAG chain — retriever → prompt → Gemini 2.0 Flash → parser |
| `app.ingestion` | `pipeline.py` | PDF → chunk → embed → persist to ChromaDB |
| `app.retrieval` | `retriever.py` | MMR retriever over the persisted ChromaDB collection |
| `app.summarization` | `summarizer.py` | Map-reduce chapter summarisation chain |
| `app.evaluation` | `metrics.py` | RAGAS (faithfulness, relevancy, recall) + ROUGE-L scoring |
| `app.ui` | `interface.py` | Gradio Blocks chat interface, mounted at `/ui` |

---

## API Reference

### `GET /`

Returns service status and navigation links.

### `GET /health`

Top-level liveness probe — used by Docker / load-balancer health checks. Returns `{"status": "ok"}`.

### `GET /api/health`

Auxiliary liveness probe via the API router — returns `{"status": "ok"}`.

### `GET /api/info`

Returns service version and active model names.

### `POST /rag/invoke` *(after ingestion)*

LangServe-managed RAG chain endpoint.

```json
{ "input": "Explain the Sobel edge detection operator." }
```

### `GET /docs`

Full OpenAPI / Swagger interactive documentation.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Library / Tool |
| --- | --- |
| LLM orchestration | LangChain 0.2, LangServe 0.2 |
| Primary LLM | Gemini 2.0 Flash (`langchain-google-genai`) |
| Fallback LLM | DeepSeek-R1 via Ollama |
| Primary embeddings | Google `text-embedding-004` (`langchain-google-genai`) |
| Fallback embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Vector store | ChromaDB (`langchain-chroma`) |
| PDF parsing | PyMuPDF (fitz), pdfplumber |
| LaTeX / math OCR | Nougat (`nougat-ocr`) |
| API server | FastAPI, Uvicorn |
| Chat UI | Gradio |
| Evaluation | RAGAS, ROUGE-score |
| Image processing | OpenCV, scikit-image |
| Testing | pytest |

---

## License

See [LICENSE](../LICENSE).
