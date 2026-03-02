"""
app/api/router.py
-----------------
FastAPI router definitions.  Chain-specific routes are registered in main.py
via LangServe ``add_routes``; this module holds auxiliary REST endpoints.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health", summary="Health check")
async def health() -> dict:
    """Liveness probe – returns 200 when the service is running."""
    return {"status": "ok"}


@router.get("/info", summary="Service metadata")
async def info() -> dict:
    """Returns version and model information."""
    return {
        "service": "Smart Learning Assistant",
        "version": "0.1.0",
        "models": ["gemini-2.0-flash", "deepseek-r1"],
    }
