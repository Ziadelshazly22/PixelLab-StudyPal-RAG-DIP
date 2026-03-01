# Smart Learning Assistant

A **RAG-Powered AI Tutor** for Digital Image Processing (DIP) learners.  
Grounded in the *Gonzalez & Woods* textbook and verified code documentation (e.g. OpenCV), it uses a dual-LLM strategy (Gemini 2.0 Flash / DeepSeek-R1) to provide mathematically accurate, cited, and conversational mentoring.

Built with **LangChain**, **LangServe**, **FastAPI**, and **ChromaDB** for a modular, production-ready microservice.

---

## Project Structure

```
smart-learning-assistant/
├── app/
│   ├── api/               # FastAPI routers
│   ├── chains/            # LangChain RAG chains
│   ├── ingestion/         # PDF parsing, chunking, embedding
│   ├── retrieval/         # Retriever logic
│   ├── summarization/     # Chapter summarization chains
│   ├── evaluation/        # Metrics, RAGAS scripts
│   └── ui/                # Gradio interface
├── data/
│   ├── raw/               # Raw PDFs (Gonzalez & Woods, code docs)
│   └── chroma_db/         # Persistent Chroma vector store
├── notebooks/             # Google Colab prototyping notebooks
├── tests/                 # Unit tests
├── .env.example
├── .gitignore
├── main.py                # FastAPI + LangServe entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & set up environment

```bash
git clone <repo-url>
cd smart-learning-assistant
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 3. Validate the installation

```bash
python validate_setup.py
```

### 4. Run the API server

```bash
python main.py
# or
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `CHROMA_PERSIST_DIR` | Path to the ChromaDB persistence directory |
| `COLLECTION_NAME` | ChromaDB collection name |

---

## Running Tests

```bash
pytest tests/
```

---

## License

See [LICENSE](../LICENSE).
