"""Engagement model wrapper.

Two backends:
  - "onnx": cv_model.onnx in repo root (4-class softmax: boredom, confusion,
    engagement, frustration). Trained by notebooks/CV_DALI.ipynb.
  - "vgg":  TF1 checkpoint shipped with vendor/engagement_vgg/ (binary engaged
    / disengaged). Loaded via tf.compat.v1.

If neither loads, backend=None; routes.py treats this as model_fallback=true
and forces boredom=0.0.
"""
import os
import sys
import logging
from typing import Optional

import cv2
import numpy as np

from .config import ONNX_PATH, ENGAGEMENT_VGG_DIR

log = logging.getLogger(__name__)


class EngagementModel:
    def __init__(self):
        self.backend: Optional[str] = None
        self.session = None
        self.input_name = None
        self.vgg_sess = None
        self.vgg_input = None
        self.vgg_output = None
        self.vgg_size = 48
        self._load()

    def _load(self):
        if os.path.exists(ONNX_PATH):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    ONNX_PATH, providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                self.backend = "onnx"
                log.info("Loaded cv_model.onnx")
                return
            except Exception as e:
                log.error(f"ONNX load failed: {e}")

        try:
            self._load_vgg_fallback()
            self.backend = "vgg"
            log.info("Loaded dali VGG fallback")
        except Exception as e:
            log.error(f"VGG fallback load failed: {e}")
            self.backend = None

    def _load_vgg_fallback(self):
        if ENGAGEMENT_VGG_DIR not in sys.path:
            sys.path.insert(0, ENGAGEMENT_VGG_DIR)
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        try:
            from VGG_const import SAVE_DIRECTORY, SAVE_MODEL_FILENAME, SIZE_FACE
        except Exception:
            SAVE_DIRECTORY = os.path.join(ENGAGEMENT_VGG_DIR, "model")
            SAVE_MODEL_FILENAME = "VGG_model"
            SIZE_FACE = 48
        # Resolve relative SAVE_DIRECTORY ('../model/') against vendor dir.
        if not os.path.isabs(SAVE_DIRECTORY):
            SAVE_DIRECTORY = os.path.normpath(os.path.join(ENGAGEMENT_VGG_DIR, SAVE_DIRECTORY))
        self.vgg_size = SIZE_FACE
        ckpt = os.path.join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME + ".meta")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"VGG checkpoint missing: {ckpt}")
        graph = tf.Graph()
        with graph.as_default():
            self.vgg_sess = tf.Session()
            saver = tf.train.import_meta_graph(ckpt)
            saver.restore(self.vgg_sess, os.path.join(SAVE_DIRECTORY, SAVE_MODEL_FILENAME))
            self.vgg_input = graph.get_tensor_by_name("input/X:0")
            self.vgg_output = graph.get_tensor_by_name("output/Softmax:0")

    def predict_engagement(self, face_bgr: np.ndarray) -> Optional[float]:
        if self.backend is None:
            return None
        try:
            if self.backend == "onnx":
                face = cv2.resize(face_bgr, (224, 224))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                inp = (face.astype(np.float32) / 255.0)[np.newaxis]
                probs = self.session.run(None, {self.input_name: inp})[0][0]
                return float(probs[2])  # engagement
            if self.backend == "vgg":
                gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (self.vgg_size, self.vgg_size))
                inp = gray.astype(np.float32).reshape(1, self.vgg_size, self.vgg_size, 1)
                probs = self.vgg_sess.run(self.vgg_output, {self.vgg_input: inp})[0]
                return float(probs[-1])
        except Exception as e:
            log.warning(f"Inference failed: {e}")
            return None
        return None
