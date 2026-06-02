import json

with open('notebooks/CV_DALI.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    # Update Dependencies (Cell 0)
    for i, line in enumerate(cell['source']):
        if line.startswith('!pip install'):
            cell['source'][i] = '!pip install -q "tensorflow==2.16.2" "tf2onnx>=1.17.0" onnxruntime opencv-python-headless "mediapipe==0.10.14" "protobuf<5"\n'
            
with open('notebooks/CV_DALI.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook patched with TF downgrade successfully!")
