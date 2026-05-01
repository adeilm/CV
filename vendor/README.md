# `vendor/` — third-party code

Vendored snapshots of upstream repositories. We import these as Python
modules; we do not author them. Treat as read-only when possible. If a
patch is unavoidable (e.g. Python 2 syntax), record the change in the
relevant `UPSTREAM_README.md`.

## Subdirectories

### [`mediapipe_rules/`](mediapipe_rules/)

Source: [Driver-State-Detection](https://github.com/.../Driver-State-Detection).

Used files:

- `eye_detector.py` — `EyeDetector.get_EAR`, `get_Gaze_Score`.
- `pose_estimation.py` — `HeadPoseEstimator.get_pose` (roll/pitch/yaw).
- `face_geometry.py` — PnP helpers used by `pose_estimation`.
- `utils.py` — `get_landmarks` (extracts Nx2 array from MediaPipe result),
  `rot_mat_to_euler`.

The directory is added to `sys.path` by `app/config.py` so the vendored
modules can use sibling imports (`from utils import ...`).

### [`engagement_vgg/`](engagement_vgg/)

Source: DALI Engagement-Recognition.

Kept as a fallback reference for `app/model.py` when `cv_model.onnx`
is missing:

- `VGG_const.py` — paths, image size, class names.
- `VGG_model.py` — architecture (binary engaged / disengaged via tflearn).
- `images/` — preserved figure from upstream README.

The runtime loads a TF1 checkpoint at `vendor/engagement_vgg/model/VGG_model.*`
(not committed; bring your own). If the checkpoint is absent, the service
runs without an engagement model — see [../docs/scoring.md](../docs/scoring.md).
