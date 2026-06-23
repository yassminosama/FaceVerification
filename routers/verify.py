"""
Verification Router
===================
Snapshot-based face verification against stored embeddings.

    POST /verify — Compare a snapshot to a registered candidate's embedding
"""

from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core import (
    EMBEDDINGS_FILE,
    SIMILARITY_THRESHOLD,
    VERIFICATION_LOG_FILE,
    cosine_similarity,
    extract_embedding,
    load_json,
    save_json,
)

router = APIRouter(tags=["verification"])


@router.post("/verify")
async def verify(
    candidate_id: str = Form(...),
    snapshot: UploadFile = File(...),
):
    """Verify an interview snapshot against the registered face embedding."""
    embeddings = load_json(EMBEDDINGS_FILE)

    if candidate_id not in embeddings:
        raise HTTPException(
            status_code=404,
            detail=f"Candidate '{candidate_id}' is not registered.",
        )

    snapshot_bytes = await snapshot.read()
    snapshot_embedding = extract_embedding(snapshot_bytes)

    stored_embedding = embeddings[candidate_id]
    score = cosine_similarity(stored_embedding, snapshot_embedding)
    matched = score >= SIMILARITY_THRESHOLD * 100
    label = "MATCHED" if matched else "NON-MATCHED"

    log = load_json(VERIFICATION_LOG_FILE)
    if candidate_id not in log:
        log[candidate_id] = []
    log[candidate_id].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "similarity": score,
        "matched": matched,
    })
    save_json(VERIFICATION_LOG_FILE, log)

    return {
        "candidate_id": candidate_id,
        "similarity_score": score,
        "matched": matched,
        "label": label,
    }
