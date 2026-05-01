# Data Flow

## Request lifecycle: `POST /cv/frame`

```
client                  routes.py            signals.py         model.py     tracker.py   db.py
  │                        │                    │                 │             │           │
  │ POST {meeting_id,      │                    │                 │             │           │
  │  student_id,           │                    │                 │             │           │
  │  frame_base64}         │                    │                 │             │           │
  ├───────────────────────►│                    │                 │             │           │
  │                        │ b64 decode → cv2   │                 │             │           │
  │                        │                    │                 │             │           │
  │                        │ extract_signals(frame)               │             │           │
  │                        ├───────────────────►│                 │             │           │
  │                        │   FaceMesh.process │                 │             │           │
  │                        │   EAR, gaze, pose  │                 │             │           │
  │                        │   crop_face        │                 │             │           │
  │                        │◄───────────────────┤                 │             │           │
  │                        │   {ear, gaze, roll,│                 │             │           │
  │                        │    pitch, yaw,     │                 │             │           │
  │                        │    face_crop}      │                 │             │           │
  │                        │                                                                │
  │                        │ predict_engagement(face_crop)        │             │           │
  │                        ├─────────────────────────────────────►│             │           │
  │                        │◄─────────────────────────────────────┤             │           │
  │                        │   engagement_conf                                              │
  │                        │                                                                │
  │                        │ tracker.add_ear(t, ear)              │             │           │
  │                        ├─────────────────────────────────────────────────────►          │
  │                        │ tracker.perclos(), ear_normalized()                            │
  │                        │◄─────────────────────────────────────────────────────          │
  │                        │                                                                │
  │                        │ fuse_scores(...)                     │             │           │
  │                        ├───────────────────►│                 │             │           │
  │                        │◄───────────────────┤                 │             │           │
  │                        │   attention, fatigue, boredom, fallback_flag                   │
  │                        │                                                                │
  │                        │ insert_one(event)                    │             │           │
  │                        ├─────────────────────────────────────────────────────────────►  │
  │                        │                                                                │
  │ 200 OK {scores,        │                                                                │
  │  face_detected, ...}   │                                                                │
  │◄───────────────────────┤                                                                │
```

## Aggregation: `GET /cv/aggregate/{meeting_id}/{student_id}`

```
1. cutoff = now() - 60s
2. mongo.find({meeting_id, student_id, ts >= cutoff}).project(scores)
3. mean(attention), mean(fatigue), mean(boredom)
4. return JSON with sample count + averages
```

## Aggregation: `GET /cv/aggregate/{meeting_id}/class`

```
1. cutoff = now() - 60s
2. mongo.aggregate:
     $match  {meeting_id, ts >= cutoff}
     $group  by student_id, avg each score    ← one row per student
3. mean across student rows                   ← democratic: each student
                                                weighs equally
4. return without student_ids (anonymous)
```

The two-stage average prevents one chatty student (sending more frames
than peers) from skewing the class-level number. It's also why the
endpoint is safe to expose to teachers without leaking which student
is the boring one.

## Failure modes

| Stage | Failure | Behaviour |
|---|---|---|
| `_decode` | Bad base64 / corrupt JPEG | 400 Bad Request, no event written |
| FaceMesh | No face in frame | Event written with `face_detected=false`, all signals None, scores fall back to neutral defaults (gaze→0.5, head pose→1.0, boredom→0). |
| `head_pose.get_pose` | PnP solver fails | Logged warning, head_attention defaults to 1.0 (assumed forward). |
| `predict_engagement` | Model raises | Logged warning, returns None → boredom=0, `model_fallback=true`. |
| Mongo unreachable | Insert exception | Bubbles up as 500. Service is intentionally fail-loud here so missing data is visible. |

## Throughput notes

- Single frame ≈ 30–80 ms on CPU (MediaPipe ~15 ms + ONNX MobileNetV2 ~20 ms).
- Recommend client posts at 1 Hz per student. 30 students × 1 Hz = 30 rps,
  well within a single uvicorn worker's budget.
- For >50 concurrent students per instance, scale horizontally and shard
  by `student_id`.
