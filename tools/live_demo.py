"""Local webcam demo. No FastAPI, no MongoDB.

Runs the rule engine + model engine on live webcam frames and overlays
scores on the feed. Three modes selectable via keyboard.

Usage:
    python tools/live_demo.py [--mode 1|2|3] [--camera 0]

Keys (while running):
    1   Engine 1 only  (rules)        → attention, fatigue
    2   Engine 2 only  (model)        → boredom
    3   Both fused                    → attention, fatigue, boredom
    l   Toggle face landmarks overlay
    r   Reset PERCLOS deque (clears fatigue history)
    q   Quit

Reuses app/signals.py + app/model.py + app/tracker.py. No duplicate logic.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.signals import extract_signals, fuse_scores
from app.model import EngagementModel
from app.tracker import StudentTracker

MODES = {1: "Engine 1 (rules)", 2: "Engine 2 (model)", 3: "Fused"}

COLOR_ATT = (0, 255, 255)    # cyan
COLOR_FAT = (0, 165, 255)    # orange
COLOR_BOR = (0, 0, 255)      # red
COLOR_HUD = (255, 255, 255)
COLOR_DIM = (200, 200, 200)


def draw_bar(frame, x, y, w, h, value, label, color):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)
    fill = int(w * float(np.clip(value, 0.0, 1.0)))
    if fill > 0:
        cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD, 1, cv2.LINE_AA)


def draw_landmarks(frame, landmarks):
    if landmarks is None:
        return
    h, w = frame.shape[:2]
    normalized = landmarks.max() <= 1.0
    for p in landmarks:
        x = int(p[0] * w) if normalized else int(p[0])
        y = int(p[1] * h) if normalized else int(p[1])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def draw_hud(frame, mode, model_backend, face_detected, sample_count,
             fallback, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"[{mode}] {MODES[mode]}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_HUD, 2, cv2.LINE_AA)
    cv2.putText(frame,
                f"backend: {model_backend or 'none'}   "
                f"face: {face_detected}   "
                f"perclos samples: {sample_count}   "
                f"fallback: {fallback}   "
                f"fps: {fps:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "keys: 1/2/3 mode  l landmarks  r reset  q quit",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DIM, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"cannot open camera {args.camera}")
        return

    model = EngagementModel()
    tracker = StudentTracker()
    mode = args.mode
    show_lms = True
    prev = time.perf_counter()
    fps = 0.0

    print("CV live demo")
    print(f"  mode: {mode} ({MODES[mode]})")
    print(f"  model backend: {model.backend or 'none'}")
    print("  keys: 1/2/3 mode, l landmarks, r reset, q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed; exiting")
            break

        frame = cv2.flip(frame, 1)  # mirror so head pose feels intuitive

        t = time.time()
        sig = extract_signals(frame)

        engagement = None
        if sig["face_crop"] is not None and sig["face_crop"].size > 0:
            engagement = model.predict_engagement(sig["face_crop"])

        tracker.add_ear(t, sig["ear"])
        perclos = tracker.perclos()
        ear_norm = tracker.ear_normalized()

        att, fat, bor, fallback = fuse_scores(
            sig["gaze"], sig["roll"], sig["pitch"], sig["yaw"],
            perclos, ear_norm, engagement,
        )

        if show_lms:
            draw_landmarks(frame, sig.get("landmarks"))

        # FPS
        now = time.perf_counter()
        dt = now - prev
        prev = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        draw_hud(frame, mode, model.backend, sig["face_detected"],
                 tracker.sample_count(), fallback, fps)

        y = 110
        if mode in (1, 3):
            draw_bar(frame, 10, y, 280, 16, att, "attention", COLOR_ATT)
            y += 50
            draw_bar(frame, 10, y, 280, 16, fat, "fatigue", COLOR_FAT)
            y += 50
        if mode in (2, 3):
            draw_bar(frame, 10, y, 280, 16, bor, "boredom", COLOR_BOR)

        cv2.imshow("CV live demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('l'):
            show_lms = not show_lms
        elif key == ord('r'):
            tracker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
