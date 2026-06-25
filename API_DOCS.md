# API Documentation

> **Base URL:** `https://yourusername-faceverificationapi.hf.space` (Hugging Face) or `http://localhost:8000` (local)
>
> **Interactive Docs:** [`/docs`](/docs) (Swagger UI) · [`/redoc`](/redoc) (ReDoc)

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System](#1-system)
  - [GET /health](#get-health)
- [Enrollment](#2-guided-multi-pose-enrollment)
  - [POST /enroll/start](#post-enrollstart)
  - [POST /enroll/frame/{session_id}](#post-enrollframesession_id)
  - [POST /enroll/retake/{session_id}](#post-enrollretakesession_id)
- [Verification](#3-face-verification)
  - [POST /verify](#post-verify)
- [Error Reference](#error-reference)
- [Integration Flow Diagram](#integration-flow-diagram)

---

## Architecture Overview

This service is a **stateless AI microservice** designed to be called by an external main backend. It handles only face recognition logic — no user management, no database access.

```
┌─────────────────────┐         ┌──────────────────────────────┐
│   Main Backend      │         │  Face Verification Service   │
│   (.NET / etc.)     │         │  (this API)                  │
│                     │         │                              │
│  • User management  │  HTTP   │  • Face detection            │
│  • Database (SQL)   │ ──────► │  • Embedding extraction      │
│  • Interview logic  │ ◄────── │  • Cosine similarity         │
│  • Stores embeddings│         │  • Multi-face detection      │
└─────────────────────┘         └──────────────────────────────┘
```

**Key design principles:**
- No `candidate_id` is required — the main backend manages identity mapping.
- Enrollment returns the computed `reference_embedding` directly — the main backend stores it.
- Verification accepts a `reference_embedding` + `snapshot` — no database lookup needed.

---

## 1. System

### `GET /health`

Health check endpoint. Returns service status and configuration.

**Response `200 OK`**

```json
{
  "status": "ok",
  "model": "ArcFace",
  "threshold": 0.6,
  "enrollment_poses": ["front", "left", "right", "up", "down"]
}
```

| Field | Type | Description |
|:---|:---|:---|
| `status` | string | Always `"ok"` if the service is running |
| `model` | string | Face recognition model in use |
| `threshold` | float | Similarity threshold for match decisions (0–1) |
| `enrollment_poses` | string[] | Required poses in enrollment order |

---

## 2. Guided Multi-Pose Enrollment

The enrollment process uses a **session-based** flow. The client starts a session, then submits camera frames one at a time. The server validates each frame against the expected head pose and advances the session automatically.

On completion, the computed `reference_embedding` (a 512-dimensional float array) is returned directly in the response for the main backend to store.

> [!IMPORTANT]
> Sessions expire after **15 minutes** of inactivity. Abandoned sessions are purged automatically.

### `POST /enroll/start`

Begin a new enrollment session.

**Request** — No parameters required.

**Response `200 OK`**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "required_poses": ["front", "left", "right", "up", "down"],
  "total_poses": 5,
  "message": "Session started. First pose: FRONT"
}
```

| Field | Type | Description |
|:---|:---|:---|
| `session_id` | string (UUID) | Pass this to all subsequent `/enroll` calls |
| `required_poses` | string[] | Ordered list of poses to capture |
| `total_poses` | integer | Total number of poses required |
| `message` | string | Human-readable status message |

---

### `POST /enroll/frame/{session_id}`

Submit one camera frame for the current required pose. The server validates the frame through a multi-step pipeline.

**Path Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `session_id` | string (UUID) | Session ID from `/enroll/start` |

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `frame` | file | ✅ | Image file (JPEG/PNG) from camera |

**Validation Pipeline**

```
Frame → Image Quality → Face Detection → Pose Match → Embedding Extraction → Store
```

**Response `200 OK`** — The `status` field indicates the result:

#### Status: `pose_captured`

The frame was accepted and the session advances to the next pose.

```json
{
  "status": "pose_captured",
  "captured_pose": "front",
  "poses_done": 1,
  "poses_remaining": 4,
  "next_pose": "left",
  "message": "✓ Front captured! Now: LEFT"
}
```

#### Status: `enrollment_complete`

All 5 poses have been captured. The averaged reference embedding is returned.

```json
{
  "status": "enrollment_complete",
  "reference_embedding": [0.0234, -0.0156, 0.0412, "...512 floats total"],
  "poses_captured": ["front", "left", "right", "up", "down"],
  "message": "All poses captured. Reference embedding computed successfully."
}
```

> [!IMPORTANT]
> The `reference_embedding` is a **512-dimensional float array**. Your main backend must store this array in its database, associated with the candidate. Pass it back to `/verify` during interview verification.

#### Status: `wrong_pose`

A face was detected but the head orientation doesn't match the required pose. **No retake is charged** — the user just needs to adjust.

```json
{
  "status": "wrong_pose",
  "target_pose": "left",
  "detected_pose": "front",
  "angles": { "yaw": -5.2, "pitch": 3.1 },
  "retakes_used": 0,
  "retakes_remaining": 5,
  "message": "Please turn your head LEFT. Currently detected: front."
}
```

#### Status: `no_face`

No face was detected in the frame. **1 retake is charged.**

```json
{
  "status": "no_face",
  "target_pose": "front",
  "retakes_used": 1,
  "retakes_remaining": 4,
  "reason": "No face detected. Ensure your face is fully visible."
}
```

#### Status: `invalid_quality`

The image failed quality checks (too small or too blurry). **1 retake is charged.**

```json
{
  "status": "invalid_quality",
  "target_pose": "front",
  "retakes_used": 1,
  "retakes_remaining": 4,
  "reason": "Image is too blurry (variance=12.3, minimum=30.0)."
}
```

#### Status: `extraction_failed`

Face was detected but the ArcFace model could not extract an embedding. **1 retake is charged.**

```json
{
  "status": "extraction_failed",
  "target_pose": "front",
  "retakes_used": 1,
  "retakes_remaining": 4,
  "reason": "Face recognition model error: ..."
}
```

> [!WARNING]
> Each pose allows a maximum of **5 retakes** (for charged failures only). Exceeding this limit terminates the session with a `400` error.

**Error Responses**

| Status | Condition |
|:---|:---|
| `400 Bad Request` | Session already complete, or retake limit exceeded |
| `404 Not Found` | Session ID not found or expired |

---

### `POST /enroll/retake/{session_id}`

Rewind the session to re-capture a specific pose. Use this when a previously accepted frame had issues (e.g. the user blinked).

**Path Parameters**

| Parameter | Type | Description |
|:---|:---|:---|
| `session_id` | string (UUID) | Session ID from `/enroll/start` |

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `pose` | string | ✅ | Pose to retake: `front`, `left`, `right`, `up`, or `down` |

**Response `200 OK`**

```json
{
  "status": "retake_ready",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "pose": "front",
  "message": "Ready to retake pose: FRONT"
}
```

**Error Responses**

| Status | Condition |
|:---|:---|
| `400 Bad Request` | Invalid pose value, or pose not in required sequence |
| `404 Not Found` | Session ID not found or expired |

---

## 3. Face Verification

### `POST /verify`

Compare a snapshot image against a provided reference embedding. The response includes a `status` field that distinguishes between a successful single-face verification and a multi-face detection event.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|:---|:---|:---|:---|
| `reference_embedding` | string | ✅ | JSON string of the float array (e.g. `"[0.023, -0.015, ...]"`) retrieved from your database |
| `snapshot` | file | ✅ | Snapshot image (JPEG/PNG) to verify |

**Response `200 OK`** — The `status` field indicates the result:

#### Status: `success`

A single face was detected and compared against the reference.

```json
{
  "status": "success",
  "faces_detected": 1,
  "similarity_score": 78.43,
  "matched": true,
  "label": "MATCHED"
}
```

| Field | Type | Description |
|:---|:---|:---|
| `status` | string | `"success"` — single face verified |
| `faces_detected` | integer | Always `1` for this status |
| `similarity_score` | float | Cosine similarity percentage (0–100) |
| `matched` | boolean | `true` if score ≥ threshold × 100 (default threshold: 60%) |
| `label` | string | `"MATCHED"` or `"NON-MATCHED"` |

#### Status: `multiple_faces`

More than one face was detected in the snapshot. No similarity score is computed. The main backend should log this as a flagged event for interview aggregation.

```json
{
  "status": "multiple_faces",
  "faces_detected": 3,
  "matched": false,
  "message": "Multiple faces detected (3). Verification requires a single face."
}
```

| Field | Type | Description |
|:---|:---|:---|
| `status` | string | `"multiple_faces"` — flag for aggregation |
| `faces_detected` | integer | Number of faces found in the snapshot |
| `matched` | boolean | Always `false` for this status |
| `message` | string | Human-readable explanation |

> [!TIP]
> Use `faces_detected` to count and report multi-face events during interview aggregation. This is a meaningful proctoring signal (e.g., someone coaching the candidate).

**Error Responses**

| Status | Condition |
|:---|:---|
| `400 Bad Request` | Invalid `reference_embedding` JSON format |
| `422 Unprocessable Entity` | No face detected, or image cannot be decoded |
| `500 Internal Server Error` | Face recognition model error |

---

## Error Reference

All error responses follow this structure:

```json
{
  "detail": "Human-readable error description."
}
```

### Image Validation Errors (apply to enrollment frames and verification snapshots)

| Status | Detail | Cause |
|:---|:---|:---|
| `400` | `Image too small (WxH). Minimum dimension is 80px.` | Image width or height below 80px |
| `400` | `Image is too blurry (variance=X, minimum=30.0).` | Laplacian variance below threshold |
| `400` | `Multiple faces detected (N). Please upload an image with a single face.` | More than one face during **enrollment** |
| `422` | `Unable to decode the uploaded image.` | Corrupt or unsupported image format |
| `422` | `No face detected in the uploaded image.` | No face found by detector |
| `500` | `Face recognition model error: ...` | ArcFace model failure |
| `500` | `Unexpected error during face detection: ...` | Unhandled exception |

### Session Errors

| Status | Detail | Cause |
|:---|:---|:---|
| `400` | `This enrollment session is already complete.` | Frame submitted after completion |
| `400` | `Too many failed attempts (5) for pose '...'. Enrollment session terminated.` | Retake limit exceeded |
| `404` | `Enrollment session not found or expired.` | Invalid or expired session ID |

### Verification Errors

| Status | Detail | Cause |
|:---|:---|:---|
| `400` | `reference_embedding must be a valid JSON array of floats.` | Malformed embedding input |

---

## Integration Flow Diagram

```mermaid
sequenceDiagram
    participant MB as Main Backend
    participant FV as Face Verification API

    rect rgb(40, 60, 80)
        Note over MB,FV: Enrollment Phase
        MB->>FV: POST /enroll/start
        FV-->>MB: session_id, required_poses

        loop For each pose (front → left → right → up → down)
            MB->>FV: POST /enroll/frame/{session_id} (frame)
            alt Pose matches
                FV-->>MB: status: "pose_captured", next_pose
            else Wrong pose
                FV-->>MB: status: "wrong_pose" (no retake charged)
                MB->>FV: POST /enroll/frame/{session_id} (retry)
            else No face / bad quality
                FV-->>MB: status: "no_face" or "invalid_quality" (retake charged)
                MB->>FV: POST /enroll/frame/{session_id} (retry)
            end
        end

        FV-->>MB: status: "enrollment_complete", reference_embedding
        Note over MB: Store reference_embedding in DB<br/>associated with the candidate
    end

    rect rgb(40, 40, 60)
        Note over MB,FV: Verification Phase (during interview)
        MB->>MB: Retrieve reference_embedding from DB
        MB->>FV: POST /verify (reference_embedding, snapshot)
        alt Single face
            FV-->>MB: status: "success", similarity_score, matched
        else Multiple faces
            FV-->>MB: status: "multiple_faces", faces_detected
            Note over MB: Log as flagged proctoring event
        end
    end
```
