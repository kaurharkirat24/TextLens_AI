"""
TextLens AI — FastAPI application entry point.

Run with:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import ingestion


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="TextLens AI",
    description="AI-powered text intelligence platform — API layer",
    version="0.1.0",
)

# ── CORS (allow React dev server) ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(ingestion.router)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "textlens-ai"}
