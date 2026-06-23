"""
Enrollment Router
=================
Guided multi-pose enrollment endpoints.

    POST /enroll/start              — Begin a new enrollment session
    POST /enroll/frame/{session_id} — Submit one camera frame for the current pose
    POST /enroll/retake/{session_id}— Force retake of a specific pose
"""

import uuid

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core import (
    EMBEDDINGS_FILE,
    MAX_RETAKES_PER_POSE,
    REQUIRED_ENROLLMENT_POSES,
    EnrollmentSession,
    HeadPose,
    PoseCapture,
    _enrollment_sessions,
    classify_pose,
    extract_embedding,
    get_pose_angles,
    load_json,
    save_json,
    validate_image,
)

router = APIRouter(prefix="/enroll", tags=["enrollment"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_retake_limit(session: EnrollmentSession, session_id: str) -> None:
    """
    Terminate the enrollment session if the retake counter has reached
    MAX_RETAKES_PER_POSE and raise an HTTPException so the caller aborts.
    """
    if session.retake_count >= MAX_RETAKES_PER_POSE:
        pose_name = session.current_target.value if session.current_target else "unknown"
        _enrollment_sessions.pop(session_id, None)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Too many failed attempts ({MAX_RETAKES_PER_POSE}) for pose "
                f"'{pose_name}'. Enrollment session terminated. "
                "Please start a new session with /enroll/start."
            ),
        )


def _build_reference_embedding(session: EnrollmentSession) -> list[float]:
    """
    Average all valid per-pose embeddings into a single L2-normalised
    reference vector.  Only call this when session.is_complete is True.

    Raises:
        HTTPException 500 — if any required pose capture is missing
    """
    valid = [c.embedding for c in session.captures.values() if c.is_valid]
    if len(valid) < len(REQUIRED_ENROLLMENT_POSES):
        raise HTTPException(
            status_code=500,
            detail="Session marked complete but has missing pose captures.",
        )
    avg = np.mean(valid, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg.tolist()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/start")
def enroll_start(candidate_id: str = Form(...)):
    """
    Begin a guided multi-pose enrollment session.

    Creates a server-side session that tracks which poses have been
    captured so far.  The client must submit frames one at a time via
    POST /enroll/frame/{session_id}.

    Returns:
        session_id      — pass this in every subsequent /enroll call
        required_poses  — ordered list of poses the session will request
        total_poses     — total number of poses required
    """
    embeddings = load_json(EMBEDDINGS_FILE)
    if candidate_id in embeddings:
        raise HTTPException(
            status_code=409,
            detail=f"Candidate '{candidate_id}' is already registered. "
                   "Use /register/update to replace.",
        )

    session_id = str(uuid.uuid4())
    _enrollment_sessions[session_id] = EnrollmentSession(candidate_id=candidate_id)

    first_pose = REQUIRED_ENROLLMENT_POSES[0].value
    return {
        "session_id": session_id,
        "candidate_id": candidate_id,
        "required_poses": [p.value for p in REQUIRED_ENROLLMENT_POSES],
        "total_poses": len(REQUIRED_ENROLLMENT_POSES),
        "message": f"Session started. First pose: {first_pose.upper()}",
    }


@router.post("/frame/{session_id}")
async def enroll_frame(session_id: str, frame: UploadFile = File(...)):
    """
    Submit one camera frame for the current required pose.

    The server validates the frame in this order:
      1. Image quality  (size, blur)
      2. Face detection (via DeepFace / RetinaFace)
      3. Pose match     (detected pose == required pose)
      4. Retake limit   (abort session if exceeded)

    On a valid capture the session advances to the next pose.
    When all poses are captured the reference embedding is computed,
    stored, and the session is deleted.

    Status values in the response:
        "wrong_pose"         — face detected but pose doesn't match (no retake charged)
        "no_face"            — no face found (retake charged)
        "invalid_quality"    — blur / size issue (retake charged)
        "extraction_failed"  — ArcFace model error (retake charged)
        "pose_captured"      — this pose is done; see next_pose
        "enrollment_complete"— all poses captured; embedding stored
    """
    session = _enrollment_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Enrollment session not found or expired. "
                   "Start a new one with /enroll/start.",
        )

    if session.is_complete:
        raise HTTPException(
            status_code=400,
            detail="This enrollment session is already complete.",
        )

    target_pose = session.current_target  # guaranteed non-None because not is_complete
    frame_bytes = await frame.read()

    # ── Step 1: Image quality ──────────────────────────────────────────────
    try:
        cv_image = validate_image(frame_bytes)
    except HTTPException as exc:
        session.retake_count += 1
        _check_retake_limit(session, session_id)
        return JSONResponse(status_code=200, content={
            "status": "invalid_quality",
            "target_pose": target_pose.value,
            "retakes_used": session.retake_count,
            "retakes_remaining": MAX_RETAKES_PER_POSE - session.retake_count,
            "reason": exc.detail,
        })

    # ── Step 2: Pose detection ─────────────────────────────────────────────
    angles = get_pose_angles(cv_image)

    if angles is None:
        # RetinaFace found no face (or isn't installed — skip pose check)
        # Only charge a retake when we can confirm no face was present.
        session.retake_count += 1
        _check_retake_limit(session, session_id)
        return JSONResponse(status_code=200, content={
            "status": "no_face",
            "target_pose": target_pose.value,
            "retakes_used": session.retake_count,
            "retakes_remaining": MAX_RETAKES_PER_POSE - session.retake_count,
            "reason": "No face detected. Ensure your face is fully visible.",
        })

    detected_pose = classify_pose(angles)

    if detected_pose != target_pose:
        # Wrong pose — guide the user but do NOT charge a retake; they just
        # need to adjust their head position.
        return JSONResponse(status_code=200, content={
            "status": "wrong_pose",
            "target_pose": target_pose.value,
            "detected_pose": detected_pose.value if detected_pose else "unknown",
            "angles": angles,
            "retakes_used": session.retake_count,
            "retakes_remaining": MAX_RETAKES_PER_POSE - session.retake_count,
            "message": (
                f"Please turn your head {target_pose.value.upper()}. "
                f"Currently detected: "
                f"{detected_pose.value if detected_pose else 'no matching pose'}."
            ),
        })

    # ── Step 3: Extract ArcFace embedding for this pose ────────────────────
    try:
        embedding = extract_embedding(frame_bytes, require_straight=False)
    except HTTPException as exc:
        session.retake_count += 1
        _check_retake_limit(session, session_id)
        return JSONResponse(status_code=200, content={
            "status": "extraction_failed",
            "target_pose": target_pose.value,
            "retakes_used": session.retake_count,
            "retakes_remaining": MAX_RETAKES_PER_POSE - session.retake_count,
            "reason": exc.detail,
        })

    # ── Step 4: Record valid capture and advance ───────────────────────────
    session.captures[target_pose] = PoseCapture(
        pose=target_pose,
        embedding=embedding,
        is_valid=True,
    )
    session.advance()

    # ── Step 5: Finalise if all poses are complete ─────────────────────────
    if session.is_complete:
        reference_embedding = _build_reference_embedding(session)

        embeddings = load_json(EMBEDDINGS_FILE)
        embeddings[session.candidate_id] = reference_embedding
        save_json(EMBEDDINGS_FILE, embeddings)

        _enrollment_sessions.pop(session_id, None)

        return {
            "status": "enrollment_complete",
            "candidate_id": session.candidate_id,
            "poses_captured": [p.value for p in REQUIRED_ENROLLMENT_POSES],
            "message": "All poses captured. Reference embedding stored successfully.",
        }

    next_pose = session.current_target
    return {
        "status": "pose_captured",
        "captured_pose": target_pose.value,
        "poses_done": session.current_pose_index,
        "poses_remaining": len(REQUIRED_ENROLLMENT_POSES) - session.current_pose_index,
        "next_pose": next_pose.value if next_pose else None,
        "message": (
            f"\u2713 {target_pose.value.capitalize()} captured! "
            f"Now: {next_pose.value.upper() if next_pose else 'done'}"
        ),
    }


@router.post("/retake/{session_id}")
def enroll_retake(session_id: str, pose: str = Form(...)):
    """
    Rewind the enrollment session to a specific pose so it can be retaken.

    Use this when the frontend detects a problem with a previously accepted
    capture (e.g. the user blinked) and needs to redo that pose.

    The session index is rewound to the requested pose and any existing
    capture for that pose is discarded.
    """
    session = _enrollment_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Enrollment session not found or expired.",
        )

    try:
        target = HeadPose(pose)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pose '{pose}'. "
                   f"Valid values: {[p.value for p in HeadPose]}",
        )

    if target not in REQUIRED_ENROLLMENT_POSES:
        raise HTTPException(
            status_code=400,
            detail=f"Pose '{pose}' is not part of the required enrollment sequence.",
        )

    session.captures.pop(target, None)
    session.current_pose_index = REQUIRED_ENROLLMENT_POSES.index(target)
    session.retake_count = 0

    return {
        "status": "retake_ready",
        "session_id": session_id,
        "pose": pose,
        "message": f"Ready to retake pose: {pose.upper()}",
    }
