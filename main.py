"""
AI Identity Verification Module
================================
Standalone FastAPI service for guided multi-pose face enrollment
and verification using ArcFace embeddings via DeepFace.

Endpoints:
    GET  /health                        — Health check
    GET  /ui                            — Serve the live test UI
    POST /enroll/start                  — Begin a guided multi-pose enrollment session
    POST /enroll/frame/{session_id}     — Submit one camera frame for the current pose
    POST /enroll/retake/{session_id}    — Force retake of a specific pose
    POST /verify                        — Verify a snapshot against stored embedding
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core import (
    CORS_ORIGINS,
    EMBEDDINGS_FILE,
    LOCAL_DB_DIR,
    MODEL_NAME,
    REQUIRED_ENROLLMENT_POSES,
    SESSION_TTL_MINUTES,
    SIMILARITY_THRESHOLD,
    VERIFICATION_LOG_FILE,
    _enrollment_sessions,
)
from routers import enroll, verify


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------
def _purge_expired_sessions() -> None:
    """Remove enrollment sessions older than SESSION_TTL_MINUTES."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
    expired = [
        sid for sid, s in _enrollment_sessions.items()
        if s.created_at < cutoff
    ]
    for sid in expired:
        _enrollment_sessions.pop(sid, None)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown + periodic session cleanup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Startup/shutdown handler using the modern lifespan protocol."""
    # ── Startup ────────────────────────────────────────────────────────────
    LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
    if not EMBEDDINGS_FILE.exists():
        EMBEDDINGS_FILE.write_text(json.dumps({}), encoding="utf-8")
    if not VERIFICATION_LOG_FILE.exists():
        VERIFICATION_LOG_FILE.write_text(json.dumps({}), encoding="utf-8")

    # Background task: purge stale enrollment sessions every 5 minutes
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)  # 5 minutes
            _purge_expired_sessions()

    cleanup_task = asyncio.create_task(_cleanup_loop())

    yield  # ── App is running ──────────────────────────────────────────────

    # ── Shutdown ───────────────────────────────────────────────────────────
    cleanup_task.cancel()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Identity Verification",
    description="Face enrollment & verification service using ArcFace embeddings.",
    version="2.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────
app.include_router(enroll.router)
app.include_router(verify.router)


# ---------------------------------------------------------------------------
# Top-level endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "threshold": SIMILARITY_THRESHOLD,
        "enrollment_poses": [p.value for p in REQUIRED_ENROLLMENT_POSES],
    }


@app.get("/ui")
def serve_ui():
    """Serve the live test UI."""
    ui_path = Path(__file__).parent / "test_ui.html"
    return FileResponse(ui_path, media_type="text/html")
