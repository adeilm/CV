# CV Module

Real-time student attention/fatigue/boredom scoring for live classroom meetings.

## Quick start

```bash
cp .env.example .env
pip install -r requirements.txt
python cv_pipeline.py     # serves on :8005
```

## Layout

```
cv_pipeline.py          entry point (uvicorn launcher)
app/                    service code
  config.py             env + thresholds
  routes.py             FastAPI endpoints
  signals.py            MediaPipe + scoring fusion
  model.py              ONNX primary, dali VGG fallback
  tracker.py            per-student PERCLOS deque
  db.py                 Mongo cv_events
  schemas.py            Pydantic
notebooks/CV_DALI.ipynb training notebook (MobileNetV2 -> cv_model.onnx)
DOCs/                   architecture, plan, contract
oumaima/                vendored MediaPipe rule pipeline (READ-ONLY)
dali/                   vendored VGG engagement repo (READ-ONLY)
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST   | /cv/frame | Score single frame, persist event |
| GET    | /cv/aggregate/{meeting_id}/{student_id} | 60s rolling avg per student |
| GET    | /cv/aggregate/{meeting_id}/class | 60s anonymous class-level avg |
| GET    | /cv/health | Liveness + model backend |

## Scoring

```
attention = 0.6 * gaze_attention + 0.4 * head_attention
fatigue   = 0.6 * PERCLOS_60s    + 0.4 * (1 - mean_EAR / 0.35)
boredom   = 1 - P(engagement)
```

If `cv_model.onnx` missing AND VGG ckpt missing: `boredom=0`, `model_fallback=true`.

See [DOCs/README.md](DOCs/README.md) and [DOCs/implementation_plan.md](DOCs/implementation_plan.md) for full details.
