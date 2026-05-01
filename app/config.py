"""Configuration: env vars, paths, scoring thresholds.

All knobs live here so other modules import constants by name. To tune
behaviour (EAR cutoff, PERCLOS window, head-pose tolerances), edit this
file only. Env-driven values come from a `.env` file at repo root (see
`.env.example`).
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(ROOT, "cv_model.onnx")
MEDIAPIPE_RULES_DIR = os.path.join(ROOT, "vendor", "mediapipe_rules")
ENGAGEMENT_VGG_DIR = os.path.join(ROOT, "vendor", "engagement_vgg")

# Vendored MediaPipe modules use sibling-style imports (e.g. `from utils import ...`).
# Inject the directory onto sys.path so those imports resolve.
if MEDIAPIPE_RULES_DIR not in sys.path:
    sys.path.insert(0, MEDIAPIPE_RULES_DIR)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "edu_platform")
PORT = int(os.getenv("CV_PORT", "8005"))

EAR_THRESHOLD = 0.25
EAR_NORMALIZED_MAX = 0.35
PERCLOS_WINDOW_SEC = 60.0
AGG_WINDOW_SEC = 60.0
ROLL_THRESH_DEG = 30.0
PITCH_THRESH_DEG = 25.0
YAW_THRESH_DEG = 25.0
GAZE_THRESH = 0.4
