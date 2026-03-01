"""
main.py – FastAPI + LangServe entry point for the Smart Learning Assistant.
"""

from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Smart Learning Assistant",
    description=(
        "RAG-Powered AI Tutor grounded in the Gonzalez & Woods DIP textbook "
        "and verified code documentation. Powered by LangChain, LangServe, "
        "and ChromaDB."
    ),
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# LangServe route registration (add chains here as they are implemented)
# ---------------------------------------------------------------------------
# from langserve import add_routes
# from app.chains.rag_chain import rag_chain
# add_routes(app, rag_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
