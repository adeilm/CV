"""Signal extraction + score fusion.

Two responsibilities:

1. `extract_signals(frame)` — runs MediaPipe FaceMesh on a single BGR
   frame and pulls every raw signal we care about: EAR (eye aperture),
   gaze score, head pose (roll/pitch/yaw), and a 224x224 face crop for
   the engagement model. Returns a flat dict; fields are None when the
   underlying detector failed.

2. `fuse_scores(...)` — applies the three published formulas (see
   docs/scoring.md) and clips each output to [0, 1].

The vendored MediaPipe helpers (`EyeDetector`, `HeadPoseEstimator`,
`get_landmarks`) are imported via the sys.path injection in config.py.
"""
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp

from eye_detector import EyeDetector
from pose_estimation import HeadPoseEstimator
from utils import get_landmarks

from .config import GAZE_THRESH, ROLL_THRESH_DEG, PITCH_THRESH_DEG, YAW_THRESH_DEG

log = logging.getLogger(__name__)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
)
eye_det = EyeDetector(show_processing=False)
head_pose = HeadPoseEstimator(show_axis=False, camera_matrix=None, dist_coeffs=None)


def gaze_attention(gaze: Optional[float]) -> float:
    if gaze is None:
        return 0.5
    return float(np.clip(1.0 - (gaze / GAZE_THRESH), 0.0, 1.0))


def head_attention(roll, pitch, yaw) -> float:
    def _norm(v, thr):
        if v is None:
            return 1.0
        val = float(v) if np.isscalar(v) else float(np.ravel(v)[0])
        return float(np.clip(1.0 - abs(val) / thr, 0.0, 1.0))
    return (_norm(roll, ROLL_THRESH_DEG)
            + _norm(pitch, PITCH_THRESH_DEG)
            + _norm(yaw, YAW_THRESH_DEG)) / 3.0


def crop_face(frame: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    normalized = landmarks.max() <= 1.0
    xs = landmarks[:, 0] * w if normalized else landmarks[:, 0]
    ys = landmarks[:, 1] * h if normalized else landmarks[:, 1]
    x1, x2 = int(max(0, xs.min())), int(min(w, xs.max()))
    y1, y2 = int(max(0, ys.min())), int(min(h, ys.max()))
    if x2 <= x1 or y2 <= y1:
        return None
    margin = int(0.15 * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
    return frame[y1:y2, x1:x2]


def extract_signals(frame: np.ndarray) -> dict:
    """Run MediaPipe + scorers. Returns dict of raw signals + face crop."""
    out = {
        "face_detected": False,
        "ear": None, "gaze": None,
        "roll": None, "pitch": None, "yaw": None,
        "face_crop": None,
    }
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lms = face_mesh.process(rgb).multi_face_landmarks
    if not lms:
        return out

    out["face_detected"] = True
    landmarks = get_landmarks(lms)
    frame_size = (frame.shape[1], frame.shape[0])

    out["ear"] = eye_det.get_EAR(landmarks=landmarks)
    gray3 = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out["gaze"] = eye_det.get_Gaze_Score(frame=gray3, landmarks=landmarks, frame_size=frame_size)

    try:
        _, roll, pitch, yaw = head_pose.get_pose(
            frame=frame.copy(), landmarks=landmarks, frame_size=frame_size
        )
        out["roll"], out["pitch"], out["yaw"] = roll, pitch, yaw
    except Exception as e:
        log.warning(f"head pose failed: {e}")

    out["face_crop"] = crop_face(frame, landmarks)
    return out


def fuse_scores(gaze, roll, pitch, yaw, perclos, ear_norm,
                engagement_conf) -> Tuple[float, float, float, bool]:
    """Returns (attention, fatigue, boredom, model_fallback)."""
    att = 0.6 * gaze_attention(gaze) + 0.4 * head_attention(roll, pitch, yaw)
    fat = 0.6 * perclos + 0.4 * (1.0 - ear_norm)
    if engagement_conf is None:
        boredom = 0.0
        fallback = True
    else:
        boredom = 1.0 - float(engagement_conf)
        fallback = False
    return (
        float(np.clip(att, 0.0, 1.0)),
        float(np.clip(fat, 0.0, 1.0)),
        float(np.clip(boredom, 0.0, 1.0)),
        fallback,
    )
