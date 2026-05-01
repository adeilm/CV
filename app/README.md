# `app/` — service code

Python package for the CV microservice. One module per responsibility,
no cross-module business logic. Imports flow downward from `routes.py`;
nothing imports `routes.py` (except the entry point `cv_pipeline.py`).

| File | Owns |
|---|---|
| [config.py](config.py) | Env vars, thresholds, vendor sys.path injection. |
| [schemas.py](schemas.py) | Pydantic request/response models. |
| [db.py](db.py) | MongoDB client + `cv_events` indexes. |
| [tracker.py](tracker.py) | `StudentTracker` rolling EAR deque, registry. |
| [model.py](model.py) | `EngagementModel` — ONNX or VGG fallback. |
| [signals.py](signals.py) | `extract_signals(frame)` + `fuse_scores(...)`. |
| [routes.py](routes.py) | FastAPI app + `/cv/*` endpoints. |

## Adding a new score

1. Add raw signal extraction in `signals.extract_signals` (return in dict).
2. Add formula to `signals.fuse_scores` (clip to [0,1]).
3. Add field to `schemas.FrameResponse`.
4. Persist in `routes.process_frame` event document.
5. Optionally aggregate: add `$avg` in `aggregate_class` pipeline.

## Adding a new endpoint

Open `routes.py`, write a handler decorated with `@app.get/post`. Keep it
thin — call into existing modules, don't inline logic.

## Adding a new threshold

Add the constant to `config.py` only. Import where needed. Document the
default in [../docs/scoring.md](../docs/scoring.md).
