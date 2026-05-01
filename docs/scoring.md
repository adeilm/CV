# Scoring

All output scores are **floats in [0, 1]**. Higher = more of that thing.

## Formulas

```
attention = 0.6 * gaze_attention + 0.4 * head_attention
fatigue   = 0.6 * perclos        + 0.4 * (1 - ear_normalized)
boredom   = 1 - engagement_confidence              (else 0 + model_fallback flag)
```

All three are clamped to [0, 1] after fusion.

## Inputs

### `gaze_attention` — derived from iris-to-eye-center distance
```
gaze_attention = clip(1 - gaze / GAZE_THRESH, 0, 1)
```
- `gaze` = normalized euclidean distance from iris center to eye center
  (vendor/mediapipe_rules/eye_detector.py).
- `GAZE_THRESH = 0.4` (config.py). Above this, considered fully off-screen.
- If MediaPipe didn't return a usable iris, `gaze` is None → defaults to 0.5
  (neutral).

### `head_attention` — mean of three normalized angles
```
head_attention = mean( norm(roll, 30°), norm(pitch, 25°), norm(yaw, 25°) )
norm(angle, threshold) = clip(1 - |angle| / threshold, 0, 1)
```
- 0° on all axes = full attention. Beyond threshold on any single axis,
  that axis contributes 0.
- Thresholds in config.py: `ROLL_THRESH_DEG=30`, `PITCH_THRESH_DEG=25`,
  `YAW_THRESH_DEG=25`.

### `perclos` — fatigue from rolling EAR
```
perclos = (# samples with ear < EAR_THRESHOLD) / total samples in last 60s
```
- `EAR_THRESHOLD = 0.25` (config.py). Below this, eye considered closed.
- `PERCLOS_WINDOW_SEC = 60.0`.
- Empty deque → 0.0.

### `ear_normalized` — eye openness scaled to [0, 1]
```
ear_normalized = clip(mean_ear_in_window / EAR_NORMALIZED_MAX, 0, 1)
```
- `EAR_NORMALIZED_MAX = 0.35` (a fully open eye).
- Empty deque → 1.0 (assume awake when no data).

### `engagement_confidence` — model output
- ONNX backend: `softmax_output[2]` (engagement class index, alphabetical order:
  boredom=0, confusion=1, engagement=2, frustration=3).
- VGG backend: `softmax_output[-1]` (binary, P(engaged)).
- Returns None on inference failure.

## Why these weights?

- 60/40 splits put primary signal first, secondary as confirmation.
- For attention, gaze is more direct (could fake forward head while looking
  at phone), so it leads.
- For fatigue, PERCLOS is the literature-validated metric; instantaneous EAR
  is nervous signal (people blink), so it weighs less.
- Boredom is direct from the model; no fusion needed.

These coefficients are starting points. Tune on real classroom data.

## Edge cases

| Situation | Behaviour |
|---|---|
| No face detected | All raw signals None. `attention` ≈ 0.7 (default head_attention=1.0, gaze_attention=0.5 → 0.3+0.4=0.7), `fatigue` from previous samples in deque (or 0 if empty), `boredom=0`, `model_fallback=true`. Event has `face_detected=false`. |
| Model not loaded (no .onnx, no VGG ckpt) | `boredom=0`, `model_fallback=true` on every frame. Attention + fatigue still work. |
| Student just joined (empty deque) | First frames have `perclos=0`, `ear_normalized=1.0` → `fatigue=0`. Realistic warm-up. |
| Continuous closed eyes for 60s | `perclos→1.0`, `ear_normalized→0` → `fatigue→1.0`. |

## Tuning

The thresholds in config.py are the primary tuning surface:

```python
EAR_THRESHOLD       = 0.25   # eye-closed cutoff
EAR_NORMALIZED_MAX  = 0.35   # fully-open EAR
PERCLOS_WINDOW_SEC  = 60.0   # rolling window
ROLL/PITCH/YAW_THRESH_DEG    # attention head-pose tolerance
GAZE_THRESH         = 0.4    # off-screen gaze cutoff
```

Change here, restart service, rerun. No callers reference these directly.
