"""
Shared constants, models, and helper functions for the AI Identity
Verification service.

This module is imported by both the enrollment and verification routers
and by ``main.py`` itself.  It contains nothing FastAPI-specific (no
endpoints) — only pure logic.
"""

import io
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Fix for Windows cp1256 encoding — DeepFace logger uses emoji that crash
# on non-UTF-8 consoles.  Must be set before importing DeepFace.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
from deepface import DeepFace
from fastapi import HTTPException
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
SIMILARITY_THRESHOLD = 0.60   # 60 %
MIN_IMAGE_DIMENSION = 80      # pixels – reject if either side is smaller
MIN_LAPLACIAN_VARIANCE = 30.0 # blur detection threshold
SESSION_TTL_MINUTES = 15      # enrollment sessions expire after this
MAX_RETAKES_PER_POSE = 5

LOCAL_DB_DIR = Path(__file__).parent / "local_db"
EMBEDDINGS_FILE = LOCAL_DB_DIR / "embeddings.json"
VERIFICATION_LOG_FILE = LOCAL_DB_DIR / "verification_log.json"

# Thread lock for JSON file read/write operations
_json_lock = threading.Lock()

# CORS origins — defaults to ["*"] for development.
# Set CORS_ORIGINS env var to a comma-separated list for production.
CORS_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "*").split(",")
    if o.strip()
]

# ---------------------------------------------------------------------------
# Head Pose Definitions
# ---------------------------------------------------------------------------

class HeadPose(Enum):
    FRONT = "front"
    LEFT  = "left"
    RIGHT = "right"
    UP    = "up"
    DOWN  = "down"


# Yaw/pitch angle bands (degrees) per required pose.
# Yaw  > 0 → face turned right; Yaw  < 0 → face turned left.
# Pitch > 0 → face tilted down; Pitch < 0 → face tilted up.
POSE_ANGLE_BANDS: dict[HeadPose, dict] = {
    HeadPose.FRONT: {"yaw": (-15,  15), "pitch": (-15,  15)},
    HeadPose.LEFT:  {"yaw": (-50, -25), "pitch": (-15,  15)},
    HeadPose.RIGHT: {"yaw": ( 25,  50), "pitch": (-15,  15)},
    HeadPose.UP:    {"yaw": (-15,  15), "pitch": (-45, -20)},
    HeadPose.DOWN:  {"yaw": (-15,  15), "pitch": ( 20,  45)},
}

# Ordered sequence of poses the enrollment session will walk through
REQUIRED_ENROLLMENT_POSES: list[HeadPose] = [
    HeadPose.FRONT,
    HeadPose.LEFT,
    HeadPose.RIGHT,
    HeadPose.UP,
    HeadPose.DOWN,
]


# ---------------------------------------------------------------------------
# Enrollment Session State
# ---------------------------------------------------------------------------

@dataclass
class PoseCapture:
    pose: HeadPose
    embedding: list[float]
    is_valid: bool = True


@dataclass
class EnrollmentSession:
    candidate_id: str
    captures: dict[HeadPose, PoseCapture] = field(default_factory=dict)
    current_pose_index: int = 0
    retake_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def current_target(self) -> HeadPose | None:
        if self.current_pose_index < len(REQUIRED_ENROLLMENT_POSES):
            return REQUIRED_ENROLLMENT_POSES[self.current_pose_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_pose_index >= len(REQUIRED_ENROLLMENT_POSES)

    def advance(self) -> None:
        """Mark current pose as done and move to the next one."""
        self.current_pose_index += 1
        self.retake_count = 0


# In-memory session store.
_enrollment_sessions: dict[str, EnrollmentSession] = {}


# ---------------------------------------------------------------------------
# Persistence helpers (thread-safe)
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    """Load a JSON file and return its contents as a dict."""
    with _json_lock:
        return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    """Write *data* to a JSON file (pretty-printed)."""
    with _json_lock:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------
def validate_image(image_bytes: bytes) -> np.ndarray:
    """
    Validate an uploaded image for minimum size and sharpness.

    Returns the decoded image as a BGR numpy array (OpenCV format).

    Raises:
        HTTPException 400  — image too small or too blurry
        HTTPException 422  — image cannot be decoded
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Unable to decode the uploaded image.")

    width, height = pil_image.size

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({width}x{height}). "
                   f"Minimum dimension is {MIN_IMAGE_DIMENSION}px.",
        )

    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < MIN_LAPLACIAN_VARIANCE:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too blurry (variance={laplacian_var:.1f}, "
                   f"minimum={MIN_LAPLACIAN_VARIANCE}).",
        )

    return cv_image


# ---------------------------------------------------------------------------
# Head Pose Detection
# ---------------------------------------------------------------------------

def get_pose_angles(cv_image: np.ndarray) -> dict[str, float] | None:
    """
    Estimate yaw and pitch from RetinaFace 5-point landmarks.

    Uses the geometric relationship between the eye midpoint, nose tip,
    and mouth midpoint — no 3-D model required.

    Returns {"yaw": float, "pitch": float} in approximate degrees, or
    None when no face is detected.

    Convention (matches POSE_ANGLE_BANDS above):
        yaw  > 0  →  face turned RIGHT
        yaw  < 0  →  face turned LEFT
        pitch > 0 →  face tilted DOWN  (nose below eye-mouth midpoint)
        pitch < 0 →  face tilted UP
    """
    try:
        # RetinaFace is already a dependency of DeepFace; import lazily so
        # the rest of the module loads even if retina-face isn't installed yet.
        from retinaface import RetinaFace  # pip install retina-face
    except ImportError:
        # Graceful fallback: skip angle-based pose check
        return None

    faces = RetinaFace.detect_faces(cv_image)
    if not faces or not isinstance(faces, dict):
        return None

    # Pick the face with the highest detection confidence
    face = max(faces.values(), key=lambda f: f.get("score", 0))
    lm = face.get("landmarks", {})

    # RetinaFace landmark keys (note: names are from the sitter's perspective)
    right_eye   = np.array(lm.get("right_eye",   [0, 0]), dtype=float)
    left_eye    = np.array(lm.get("left_eye",    [0, 0]), dtype=float)
    nose        = np.array(lm.get("nose",        [0, 0]), dtype=float)
    mouth_right = np.array(lm.get("mouth_right", [0, 0]), dtype=float)
    mouth_left  = np.array(lm.get("mouth_left",  [0, 0]), dtype=float)

    eye_width = float(np.linalg.norm(left_eye - right_eye))
    if eye_width < 1.0:
        return None

    eye_mid   = (right_eye + left_eye) / 2.0
    mouth_mid = (mouth_right + mouth_left) / 2.0
    face_height = float(mouth_mid[1] - eye_mid[1])
    if face_height < 1.0:
        return None

    # --- Yaw estimation ---------------------------------------------------
    # The nose tip shifts horizontally away from the eye-midpoint as the
    # head rotates.  Normalise by inter-eye distance to be scale-invariant.
    nose_offset_x = (nose[0] - eye_mid[0]) / eye_width
    # Scale factor of 90 maps the ±1 normalised range to ±90°.
    yaw = float(nose_offset_x * 90.0)

    # --- Pitch estimation -------------------------------------------------
    # nose_offset_y == 0.5 when the nose is exactly half-way between eyes
    # and mouth (frontal).  Values below 0.5 mean the nose is close to the
    # eyes → head tilted up (negative pitch).
    nose_offset_y = (nose[1] - eye_mid[1]) / face_height
    pitch = float((nose_offset_y - 0.5) * 90.0)

    return {"yaw": yaw, "pitch": pitch}


def classify_pose(angles: dict[str, float]) -> HeadPose | None:
    """
    Map a (yaw, pitch) angle dict onto one of the required HeadPose values.
    Returns None when no band matches.
    """
    for pose, band in POSE_ANGLE_BANDS.items():
        yaw_ok   = band["yaw"][0]   <= angles["yaw"]   <= band["yaw"][1]
        pitch_ok = band["pitch"][0] <= angles["pitch"] <= band["pitch"][1]
        if yaw_ok and pitch_ok:
            return pose
    return None


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embedding(image_bytes: bytes, require_straight: bool = False) -> list[float]:
    """
    Validate the image, detect exactly one face, and return a 512-d
    ArcFace embedding vector.

    Args:
        image_bytes:      Raw bytes of the uploaded image.
        require_straight: When True, also verify the face is looking
                          straight ahead.

    Raises:
        HTTPException 400  — multiple faces, image quality, or bad pose
        HTTPException 422  — no face detected / image cannot be decoded
        HTTPException 500  — model error
    """
    cv_image = validate_image(image_bytes)

    try:
        results: list[dict[str, Any]] = DeepFace.represent(
            img_path=cv_image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError as exc:
        exc_msg = str(exc).lower()
        if "no face" in exc_msg or "face could not be detected" in exc_msg:
            raise HTTPException(
                status_code=422,
                detail="No face detected in the uploaded image.",
            )
        raise HTTPException(
            status_code=500,
            detail=f"Face recognition model error: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during face detection: {exc}",
        )

    if len(results) == 0:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the uploaded image.",
        )

    if len(results) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple faces detected ({len(results)}). "
                   "Please upload an image with a single face.",
        )

    if require_straight:
        angles = get_pose_angles(cv_image)
        if angles is not None:
            detected = classify_pose(angles)
            if detected != HeadPose.FRONT:
                direction = detected.value.upper() if detected else "unknown direction"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid head pose (facing {direction}). "
                        "Please look straight at the camera and retry."
                    ),
                )

    return results[0]["embedding"]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Return cosine similarity as a percentage (0–100)."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    similarity = dot / norm
    return round(float(max(0.0, min(1.0, similarity))) * 100, 2)
