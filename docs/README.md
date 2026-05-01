# CV Module — Documentation Index

This is the computer-vision microservice of the EduAI platform. It scores
student webcam frames in real time on three axes: **attention**, **fatigue**,
**boredom**. Designed to be polled by the teacher dashboard at ~1 Hz per
student.

## Read these in order

1. [architecture.md](architecture.md) — components, layered design, why two
   "engines" (rule-based + learned).
2. [data_flow.md](data_flow.md) — what happens to a single frame, end to end.
3. [scoring.md](scoring.md) — exact formulas, thresholds, edge cases.
4. [implementation_plan.md](implementation_plan.md) — original P-CV1 plan for
   the ONNX engagement model (training-time reference).

## Quick links

- API surface: see root [../README.md](../README.md) endpoints table.
- Service entry: [../cv_pipeline.py](../cv_pipeline.py).
- Service code: [../app/](../app/) (one module per responsibility).
- Vendored 3rd-party code: [../vendor/](../vendor/).

## Glossary

| Term | Meaning |
|---|---|
| **EAR** | Eye-Aspect Ratio. Geometric measure of eye openness from 6 landmarks per eye. Drops below ~0.20 when eye closes. |
| **PERCLOS** | Percentage of time eyes are closed over a window. Standard fatigue metric. |
| **Gaze score** | Normalized euclidean distance from iris center to eye center. Higher = looking off-screen. |
| **Head pose** | Roll/pitch/yaw of head in degrees, derived from facial landmarks via PnP. |
| **Engagement** | Output of the learned classifier (ONNX or VGG fallback). |
| **Frame** | One JPEG-encoded webcam image, posted as base64 in the request body. |
| **Event** | One MongoDB document in `cv_events`, written per processed frame. |
