# `tools/` — local utilities

Standalone helpers that live outside the FastAPI service. Reuse `app/`
modules; do not duplicate logic.

## `live_demo.py`

Local webcam viewer. Opens default camera, runs MediaPipe + engagement
model on every frame, draws scores as on-screen bars. No HTTP, no
MongoDB.

```bash
python tools/live_demo.py            # mode 3 (fused) by default
python tools/live_demo.py --mode 1   # rules only
python tools/live_demo.py --mode 2   # model only
python tools/live_demo.py --camera 1 # second webcam
```

Keys at runtime:

| Key | Action |
|-----|--------|
| `1` | Engine 1 only — show attention + fatigue |
| `2` | Engine 2 only — show boredom |
| `3` | Fused — show all three |
| `l` | Toggle face-landmark dots overlay |
| `r` | Reset PERCLOS deque (clears fatigue history) |
| `q` / `Esc` | Quit |

### What you'll see

- Mirrored webcam feed (so head pose feels intuitive).
- Top HUD: current mode, model backend (`onnx` / `vgg` / `none`), face
  detected, PERCLOS sample count, model-fallback flag, FPS.
- Score bars: filled proportional to value 0 → 1, with numeric label.

### Caveats

- **Cold start.** PERCLOS uses a 60 s rolling window. Fatigue stays
  near 0 for the first minute. Use `r` to reset for repeat tests.
- **Model fallback.** If `cv_model.onnx` missing AND VGG checkpoint
  absent, `backend: none` and boredom is forced to 0 (HUD shows
  `fallback: True`). Train the notebook or drop in `cv_model.onnx`
  to enable mode 2.
- **One face only.** Demo uses MediaPipe single-face mode (matches the
  service).
