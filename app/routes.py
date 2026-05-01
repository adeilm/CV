"""FastAPI app: HTTP endpoints + per-request orchestration.

This module wires the layers together; it owns no business logic of
its own. For each `/cv/frame` it: decodes the JPEG, calls
`signals.extract_signals`, runs the engagement model on the face crop,
updates the per-student PERCLOS deque, fuses scores, and persists the
event to MongoDB. Aggregation endpoints query MongoDB directly.

See docs/data_flow.md for a step-by-step trace.
"""
import time
import base64
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException

from .config import AGG_WINDOW_SEC
from .schemas import FrameRequest, FrameResponse
from .db import events_col
from .model import EngagementModel
from .tracker import get_tracker
from .signals import extract_signals, fuse_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cv_pipeline")

app = FastAPI(title="CV Pipeline", version="1.0")
model = EngagementModel()


def _decode(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("decode failed")
    return frame


@app.post("/cv/frame", response_model=FrameResponse)
def process_frame(req: FrameRequest):
    # TODO: add JWT auth here (verify token, extract user_id, check student_id matches).
    try:
        frame = _decode(req.frame_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad frame: {e}")

    t_now = time.time()
    tracker = get_tracker(req.student_id)
    sig = extract_signals(frame)

    engagement_conf: Optional[float] = None
    if sig["face_crop"] is not None and sig["face_crop"].size > 0:
        engagement_conf = model.predict_engagement(sig["face_crop"])

    tracker.add_ear(t_now, sig["ear"])
    perclos = tracker.perclos()
    ear_norm = tracker.ear_normalized()

    att, fat, bor, fallback = fuse_scores(
        sig["gaze"], sig["roll"], sig["pitch"], sig["yaw"],
        perclos, ear_norm, engagement_conf,
    )

    events_col.insert_one({
        "meeting_id": req.meeting_id,
        "student_id": req.student_id,
        "ts": t_now,
        "attention": att,
        "fatigue": fat,
        "boredom": bor,
        "ear": sig["ear"],
        "gaze": sig["gaze"],
        "perclos": perclos,
        "engagement_conf": engagement_conf,
        "face_detected": sig["face_detected"],
        "model_fallback": fallback,
        "backend": model.backend,
    })

    return FrameResponse(
        attention=att, fatigue=fat, boredom=bor,
        face_detected=sig["face_detected"],
        model_fallback=fallback, backend=model.backend,
    )


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


@app.get("/cv/aggregate/{meeting_id}/{student_id}")
def aggregate_student(meeting_id: str, student_id: str):
    cutoff = time.time() - AGG_WINDOW_SEC
    rows = list(events_col.find(
        {"meeting_id": meeting_id, "student_id": student_id, "ts": {"$gte": cutoff}},
        {"_id": 0, "attention": 1, "fatigue": 1, "boredom": 1},
    ))
    return {
        "meeting_id": meeting_id,
        "student_id": student_id,
        "window_sec": AGG_WINDOW_SEC,
        "samples": len(rows),
        "attention": _avg(rows, "attention"),
        "fatigue": _avg(rows, "fatigue"),
        "boredom": _avg(rows, "boredom"),
    }


@app.get("/cv/aggregate/{meeting_id}/class")
def aggregate_class(meeting_id: str):
    cutoff = time.time() - AGG_WINDOW_SEC
    pipeline = [
        {"$match": {"meeting_id": meeting_id, "ts": {"$gte": cutoff}}},
        {"$group": {
            "_id": "$student_id",
            "attention": {"$avg": "$attention"},
            "fatigue": {"$avg": "$fatigue"},
            "boredom": {"$avg": "$boredom"},
        }},
    ]
    per_student = list(events_col.aggregate(pipeline))
    if not per_student:
        return {"meeting_id": meeting_id, "students": 0,
                "attention": None, "fatigue": None, "boredom": None}
    return {
        "meeting_id": meeting_id,
        "window_sec": AGG_WINDOW_SEC,
        "students": len(per_student),
        "attention": float(np.mean([s["attention"] for s in per_student])),
        "fatigue": float(np.mean([s["fatigue"] for s in per_student])),
        "boredom": float(np.mean([s["boredom"] for s in per_student])),
    }


@app.get("/cv/health")
def health():
    return {"status": "ok", "backend": model.backend}
