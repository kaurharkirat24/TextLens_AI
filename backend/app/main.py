"""
TextLens AI — FastAPI application entry point.

Run with:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import ingestion, analysis, semantic


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("textlens.api")


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
app.include_router(analysis.router)
app.include_router(semantic.router)


# ── Health check ──────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Request started %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info(
        "Request completed %s %s status=%s",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "textlens-ai"}
