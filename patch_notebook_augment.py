import json

with open('notebooks/CV_DALI.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    # Fix Cell 4 — Data Pipeline
    # The problem: augmentation (RandomBrightness) is applied AFTER rescaling to [0,1],
    # which creates negative pixel values. MobileNetV2's internal rescaling then
    # pushes them even further negative. The model sees garbage.
    # Fix: Apply augmentation BEFORE rescaling, when pixels are still in [0, 255].
    if any('image_dataset_from_directory' in line for line in cell['source']):
        cell['source'] = [
            "import tensorflow as tf\n",
            "from sklearn.utils.class_weight import compute_class_weight\n",
            "\n",
            "IMG_SIZE = (224, 224)\n",
            "BATCH_SIZE = 32\n",
            "\n",
            "train_ds = tf.keras.utils.image_dataset_from_directory(\n",
            "    'frames/Train', image_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
            "    label_mode='categorical', shuffle=True, seed=42)\n",
            "\n",
            "val_ds = tf.keras.utils.image_dataset_from_directory(\n",
            "    'frames/Validation', image_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
            "    label_mode='categorical', shuffle=False)\n",
            "\n",
            "test_ds = tf.keras.utils.image_dataset_from_directory(\n",
            "    'frames/Test', image_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
            "    label_mode='categorical', shuffle=False)\n",
            "\n",
            "print(f\"Classes: {train_ds.class_names}\")\n",
            "\n",
            "# Augmentation FIRST (on raw [0, 255] uint8 images), THEN rescale.\n",
            "# RandomBrightness and RandomContrast work correctly on [0, 255] range.\n",
            "augment = tf.keras.Sequential([\n",
            "    tf.keras.layers.RandomFlip('horizontal'),\n",
            "    tf.keras.layers.RandomBrightness(0.2),\n",
            "    tf.keras.layers.RandomContrast(0.2),\n",
            "])\n",
            "\n",
            "# Rescale to [0, 1] AFTER augmentation\n",
            "rescale = tf.keras.layers.Rescaling(1.0 / 255)\n",
            "\n",
            "train_ds = train_ds.map(lambda x, y: (rescale(augment(x, training=True)), y))\n",
            "val_ds   = val_ds.map(lambda x, y: (rescale(x), y))\n",
            "test_ds  = test_ds.map(lambda x, y: (rescale(x), y))\n",
            "\n",
            "# Prefetch\n",
            "AUTOTUNE = tf.data.AUTOTUNE\n",
            "train_ds = train_ds.prefetch(AUTOTUNE)\n",
            "val_ds   = val_ds.prefetch(AUTOTUNE)\n",
            "test_ds  = test_ds.prefetch(AUTOTUNE)\n",
            "\n",
            "# Compute class weights\n",
            "train_labels = []\n",
            "for _, labels in tf.keras.utils.image_dataset_from_directory(\n",
            "    'frames/Train', image_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
            "    label_mode='int', shuffle=False):\n",
            "    train_labels.extend(labels.numpy())\n",
            "train_labels = np.array(train_labels)\n",
            "\n",
            "weights = compute_class_weight('balanced', classes=np.arange(4), y=train_labels)\n",
            "class_weight = {i: w for i, w in enumerate(weights)}\n",
            "print(f\"Class weights: {class_weight}\")\n",
            "\n",
            "# Sample visualization\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "plt.figure(figsize=(12, 6))\n",
            "for images, labels in train_ds.take(1):\n",
            "    for i in range(min(16, len(images))):\n",
            "        ax = plt.subplot(4, 4, i + 1)\n",
            "        plt.imshow(images[i].numpy().clip(0, 1))\n",
            "        cls_idx = tf.argmax(labels[i]).numpy()\n",
            "        plt.title(CLASS_NAMES[cls_idx], fontsize=9)\n",
            "        plt.axis('off')\n",
            "plt.suptitle('Training Samples (augmented)')\n",
            "plt.tight_layout()\n",
            "plt.show()",
        ]

with open('notebooks/CV_DALI.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook Cell 4 fixed: augmentation now applied BEFORE rescaling!")
