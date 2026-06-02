import json

with open('notebooks/CV_DALI.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    # 1. Match Dependencies (Cell 0)
    for i, line in enumerate(cell['source']):
        if '!pip install -q "tf2onnx>=1.17.0" onnxruntime opencv-python-headless\n' == line:
            cell['source'][i] = '!pip install -q "tf2onnx>=1.17.0" onnxruntime opencv-python-headless "mediapipe==0.10.14"\n'
            
    # 2. Fix Training/Inference Mismatch (Cell 3)
    if any('haarcascade_frontalface_default.xml' in line for line in cell['source']):
        new_source = [
            "import cv2\n",
            "from tqdm import tqdm\n",
            "import mediapipe as mp\n",
            "import numpy as np\n",
            "import os\n",
            "\n",
            "face_mesh = mp.solutions.face_mesh.FaceMesh(\n",
            "    static_image_mode=False,\n",
            "    min_detection_confidence=0.5,\n",
            "    min_tracking_confidence=0.5,\n",
            "    refine_landmarks=True,\n",
            ")\n",
            "\n",
            "def crop_face(frame, landmarks):\n",
            "    h, w = frame.shape[:2]\n",
            "    normalized = landmarks.max() <= 1.0\n",
            "    xs = landmarks[:, 0] * w if normalized else landmarks[:, 0]\n",
            "    ys = landmarks[:, 1] * h if normalized else landmarks[:, 1]\n",
            "    x1, x2 = int(max(0, xs.min())), int(min(w, xs.max()))\n",
            "    y1, y2 = int(max(0, ys.min())), int(min(h, ys.max()))\n",
            "    if x2 <= x1 or y2 <= y1:\n",
            "        return None\n",
            "    margin = int(0.15 * max(x2 - x1, y2 - y1))\n",
            "    x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)\n",
            "    x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)\n",
            "    return frame[y1:y2, x1:x2]\n",
            "\n",
            "FRAME_INTERVAL = 15\n",
            "OUTPUT_DIR = 'frames'\n",
            "\n",
            "def extract_faces_from_clip(video_path, output_folder, clip_id, interval=15):\n",
            "    os.makedirs(output_folder, exist_ok=True)\n",
            "    cap = cv2.VideoCapture(video_path)\n",
            "    count = 0\n",
            "    saved = 0\n",
            "    while True:\n",
            "        ret, frame = cap.read()\n",
            "        if not ret:\n",
            "            break\n",
            "        if count % interval == 0:\n",
            "            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
            "            results = face_mesh.process(rgb)\n",
            "            if results.multi_face_landmarks:\n",
            "                lm_list = results.multi_face_landmarks[0].landmark\n",
            "                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in lm_list])\n",
            "                roi = crop_face(frame, landmarks)\n",
            "                if roi is not None and roi.size > 0 and roi.shape[0] >= 2 and roi.shape[1] >= 2:\n",
            "                    face = cv2.resize(roi, (224, 224))\n",
            "                    fname = f\"{clip_id}_f{count}.jpg\"\n",
            "                    cv2.imwrite(os.path.join(output_folder, fname), face)\n",
            "                    saved += 1\n",
            "        count += 1\n",
            "    cap.release()\n",
            "    return saved\n",
        ]
        
        # Replace up to `    return saved\n`
        end_idx = 0
        for i, line in enumerate(cell['source']):
            if line == "    return saved\n":
                end_idx = i
                break
        if end_idx > 0:
            cell['source'] = new_source + cell['source'][end_idx+1:]
            
    # 3. Improve Fine-Tuning Depth (Cell 7)
    for i, line in enumerate(cell['source']):
        if "for layer in base_model.layers[:-3]:\n" == line:
            cell['source'][i] = "for layer in base_model.layers[:-30]:\n"

with open('notebooks/CV_DALI.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook patched successfully!")
