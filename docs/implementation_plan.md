# P-CV1: 4-Class Affective State Model → `cv_model.onnx`

> Training-time reference. Not consumed by the runtime service except via
> the produced `cv_model.onnx` file. See [scoring.md](scoring.md) and
> [architecture.md](architecture.md) for service-side behaviour.

## Goal

Produce a single ONNX file (`cv_model.onnx`) that takes a `[1, 224, 224, 3]`
float32 face crop and outputs `[1, 4]` float32 softmax probabilities for
**[boredom, confusion, engagement, frustration]** (alphabetical order).

This file plugs into the runtime service via `onnxruntime` — no TensorFlow
at inference time.

## Pipeline

1. **Download DAiSEE** — 15 GB, 9,068 clips × ~10 s. Already in
   `notebooks/CV_DALI.ipynb` via `gdown`.
2. **Parse labels** — DAiSEE provides 0–3 ratings on 4 affective columns
   per clip. Convert to single-label by `argmax` across columns. Ties
   prefer engagement.
3. **Extract face crops** — every 15th frame, OpenCV Haar cascade, largest
   face, 15% margin, resize to 224×224, save as JPEG into ImageFolder
   layout under `frames/{Train,Validation,Test}/{boredom,confusion,engagement,frustration}/`.
4. **Build dataset** — `tf.keras.utils.image_dataset_from_directory`,
   `Rescaling(1/255)` inside the model, train-time augmentation
   (RandomFlip, RandomBrightness, RandomContrast).
5. **Class weights** — `sklearn.utils.compute_class_weight('balanced')`
   to combat engagement majority.
6. **Architecture** — MobileNetV2 (ImageNet, no top) → GAP → Dense(128, relu)
   → Dense(4, softmax).
7. **Training** —
   - Phase 1: base frozen, lr=1e-3, 15 epochs, EarlyStopping on val_acc.
   - Phase 2: unfreeze last 3 layers, lr=1e-4, 15 more epochs.
8. **Evaluate** — confusion matrix, per-class F1, on test split.
9. **Export** — `tf2onnx.convert.from_keras` opset 15, write `cv_model.onnx`.
10. **Verify** — file < 20 MB, output shape (1, 4), softmax sums to ~1.0,
    Keras-vs-ONNX parity within 1e-4.

## Model card (consumer reference)

| Property | Value |
|---|---|
| File | `cv_model.onnx` (drop in repo root) |
| Input name | `input` |
| Input shape | `[1, 224, 224, 3]` float32, RGB, normalized to [0, 1] |
| Output shape | `[1, 4]` float32 softmax |
| Output order | `[boredom, confusion, engagement, frustration]` |
| Service usage | `engagement_confidence = output[0][2]` |
| Framework | ONNX opset 15, loadable with `onnxruntime` |
| Size | ~9 MB (MobileNetV2 base) |

## FER2013 fallback (only if DAiSEE unavailable)

Maps FER2013 emotions → 4 affective classes:

| FER label | Mapped to |
|---|---|
| angry, disgust | frustration |
| fear, surprise | confusion |
| happy, neutral | engagement |
| sad | boredom |

> **Warning.** Approximate. Real model must train on DAiSEE for
> meaningful engagement detection.

## Verification

Automated (in notebook):
1. Forward smoke: random tensor → shape (1, 4), sum ≈ 1.
2. ONNX parity: 5 test images, max diff < 1e-4.
3. File size < 20 MB.
4. Per-class F1 on test split.
5. Test accuracy > 25 % (random baseline).

Manual:
1. Visual spot check: 16 random test images with predicted vs. ground
   truth labels.
2. Service smoke: `curl /cv/health` returns `backend: "onnx"` after
   `cv_model.onnx` is dropped in.
