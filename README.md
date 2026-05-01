# CV Module

Real-time student attention / fatigue / boredom scoring for live classroom
meetings. FastAPI service. MongoDB-backed.

## Quick start

```bash
cp .env.example .env             # edit MONGO_URI if needed
pip install -r requirements.txt
python cv_pipeline.py            # serves on :8005
```

Or run the local webcam demo (no Mongo, no HTTP):

```bash
python tools/live_demo.py        # see tools/README.md
```

Health check:
```bash
curl http://localhost:8005/cv/health
# { "status": "ok", "backend": "onnx" | "vgg" | null }
```

## Layout

```
cv_pipeline.py          entry point — uvicorn launcher
app/                    service code (one module per responsibility)
  README.md             ← per-module map
  config.py             env + thresholds
  routes.py             FastAPI endpoints
  signals.py            MediaPipe + scoring fusion
  model.py              ONNX primary, VGG fallback
  tracker.py            per-student PERCLOS deque
  db.py                 Mongo cv_events
  schemas.py            Pydantic
notebooks/
  CV_DALI.ipynb         training notebook → produces cv_model.onnx
tools/
  README.md             utility scripts overview
  live_demo.py          local webcam viewer (no HTTP, no Mongo)
docs/                   ← read first if new to the project
  README.md             docs index + glossary
  architecture.md       components, layered design
  data_flow.md          per-frame request lifecycle
  scoring.md            formulas, thresholds, edge cases
  implementation_plan.md training-time plan for the ONNX model
vendor/                 vendored 3rd-party code (read-only)
  README.md
  mediapipe_rules/      MediaPipe rule pipeline (was: oumaima/)
  engagement_vgg/       VGG engagement classifier (was: dali/)
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST   | `/cv/frame` | Score single frame, persist event |
| GET    | `/cv/aggregate/{meeting_id}/{student_id}` | 60 s rolling avg per student |
| GET    | `/cv/aggregate/{meeting_id}/class` | 60 s anonymous class-level avg |
| GET    | `/cv/health` | Liveness + active model backend |

## Scoring

```
attention = 0.6 · gaze_attention + 0.4 · head_attention
fatigue   = 0.6 · PERCLOS_60s    + 0.4 · (1 − mean_EAR / 0.35)
boredom   = 1 − P(engagement)        (or 0 with model_fallback flag)
```

Full formulas + thresholds: [docs/scoring.md](docs/scoring.md).

## How it works

Per `/cv/frame`: decode JPEG → MediaPipe FaceMesh → EAR / gaze / head pose
+ face crop → engagement model (ONNX or VGG) → fuse scores → write to
MongoDB → return scores. Trace: [docs/data_flow.md](docs/data_flow.md).

Per `/cv/aggregate/...`: 60 s rolling window from `cv_events`, averaged.

## New to the project?

Read in this order:

1. [docs/architecture.md](docs/architecture.md) — components + why two
   "engines" (rule-based + learned).
2. [docs/data_flow.md](docs/data_flow.md) — what happens to a single frame.
3. [docs/scoring.md](docs/scoring.md) — formulas + thresholds.
4. [app/README.md](app/README.md) — service code map.
5. [vendor/README.md](vendor/README.md) — third-party code notes.

## Operational notes

- **One uvicorn worker only.** MediaPipe and ONNX session are module-level
  singletons; not safe across worker processes. Scale by container, not
  by `--workers`.
- **JWT auth not wired yet.** `# TODO` marker at `/cv/frame` in
  `app/routes.py`. Add before any production exposure.
- **`cv_model.onnx`** — drop in repo root after running
  `notebooks/CV_DALI.ipynb`. Without it, service falls back to VGG (if
  checkpoint present) or returns `boredom=0` with `model_fallback=true`.
