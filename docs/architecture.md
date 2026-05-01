# Architecture

## High-level

```
┌─────────────────┐      POST /cv/frame       ┌──────────────────────┐
│  Student client │  ───────────────────────► │  CV service (this)   │
│  (browser, JS)  │  ◄─────────────────────── │  FastAPI :8005       │
└─────────────────┘    {attention, fatigue,   └──────────┬───────────┘
                        boredom, ...}                    │
                                                         │ insert event
                                                         ▼
┌─────────────────┐      GET /cv/aggregate    ┌──────────────────────┐
│ Teacher dash    │  ───────────────────────► │  MongoDB             │
│ (browser, JS)   │  ◄─────────────────────── │  cv_events coll.     │
└─────────────────┘    rolling 60s averages   └──────────────────────┘
```

## Two-engine design

Per frame, the service runs **two independent engines** and fuses their outputs:

1. **Rule engine** (vendor/mediapipe_rules) — pure geometry on MediaPipe
   FaceMesh landmarks. Produces:
   - EAR per eye → mean EAR
   - PERCLOS over rolling 60 s window
   - Gaze score (iris-to-eye-center distance)
   - Head pose roll/pitch/yaw via PnP
   No training, fast (~10 ms CPU), interpretable.

2. **Learned engine** (cv_model.onnx, or vendor/engagement_vgg fallback) —
   CNN classifier on the cropped face region. Produces:
   - 4-class softmax `[boredom, confusion, engagement, frustration]` (ONNX)
   - or binary engaged/disengaged (VGG fallback)

| Concern | Why split |
|---|---|
| **Robustness** | Rules don't need labeled data and degrade gracefully (eyes still close even if subject is novel). |
| **Affect** | Geometric features can't tell bored from engaged. That requires learning. |
| **Failure mode** | If the model fails to load, the rule layer still produces attention + fatigue. Boredom degrades to 0 with a `model_fallback=true` flag. |

## Layered package map

```
cv_pipeline.py            entry point, launches uvicorn
└── app/
    ├── routes.py         HTTP — orchestration only, no business logic
    │   ├── schemas.py    Pydantic request/response
    │   ├── db.py         MongoDB client + indexes
    │   ├── model.py      EngagementModel — ONNX primary, VGG fallback
    │   ├── tracker.py    StudentTracker — rolling EAR/PERCLOS per student
    │   ├── signals.py    extract_signals + fuse_scores  (← vendor calls live here)
    │   └── config.py     env vars + thresholds (single source of truth)
    └── (vendor/ imported via sys.path injection in config.py)
```

Dependency direction is one-way:

```
routes.py ─► signals, model, tracker, db, schemas, config
signals.py ─► config + vendor/mediapipe_rules
model.py   ─► config + vendor/engagement_vgg (only if ONNX missing)
tracker.py ─► config
db.py      ─► config
```

`config.py` depends on nothing (except env). All other modules depend on it.
No circular imports.

## Vendored 3rd-party code

`vendor/` holds external code we don't author. Two repos, flattened:

- `vendor/mediapipe_rules/` — extracted from
  [Driver-State-Detection](https://github.com/.../Driver-State-Detection).
  Only the four files we actually use: `eye_detector.py`,
  `pose_estimation.py`, `face_geometry.py`, `utils.py`.
- `vendor/engagement_vgg/` — extracted from the DALI Engagement-Recognition
  repo. Only `VGG_const.py` + `VGG_model.py` kept as a fallback reference.

`UPSTREAM_README.md` in each vendor dir preserves the original project's
README. Treat all of `vendor/` as read-only when possible; if you must
patch (e.g. Python 2 → 3 syntax in VGG_model.py), note it in the
`UPSTREAM_README.md` for that vendor.

## Concurrency model

- **Single uvicorn worker.** MediaPipe `FaceMesh` and the ONNX session
  are module-level singletons; not safe across multiple worker processes.
  Scale horizontally via separate container instances behind a load
  balancer, sticky on `student_id` if needed.
- **Per-student state** lives in a thread-locked dict in `tracker.py`.
  Each student's deque has its own lock; cross-student requests don't
  contend.

## Storage

`cv_events` document shape:

```json
{
  "meeting_id": "string",
  "student_id": "string",
  "ts": 1714500000.123,
  "attention": 0.0,
  "fatigue": 0.0,
  "boredom": 0.0,
  "ear": 0.27,
  "gaze": 0.18,
  "perclos": 0.04,
  "engagement_conf": 0.71,
  "face_detected": true,
  "model_fallback": false,
  "backend": "onnx"
}
```

Retention is the operator's call. MongoDB TTL index on `ts` is a fine
default; not configured by this service.
