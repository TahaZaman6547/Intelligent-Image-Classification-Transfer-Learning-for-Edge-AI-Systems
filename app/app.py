"""
Flask REST API — CIFAR-10 Image Classifier
Serves the best trained model (MobileNetV2) for Edge AI inference.
Deployable to Railway, Render, Heroku, or any cloud platform.

Endpoints:
  GET  /health   → status check
  GET  /classes  → list CIFAR-10 classes
  POST /predict  → single or batch image classification
"""

import os, io, json
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'mobilenet_best.keras')
IMG_SIZE   = int(os.environ.get('IMG_SIZE', 96))
PORT       = int(os.environ.get('PORT', 8000))

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None


def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None}), 200


@app.route('/classes', methods=['GET'])
def classes():
    return jsonify({'classes': CLASS_NAMES, 'count': len(CLASS_NAMES)}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    files = request.files.getlist('file')
    if not files:
        return jsonify({'error': 'No file uploaded. Use key "file".'}), 400

    results = []
    for f in files:
        try:
            img_array = preprocess_image(f.read())
            probs     = model.predict(img_array, verbose=0)[0].tolist()
            top_idx   = int(np.argmax(probs))
            results.append({
                'filename'        : f.filename,
                'predicted_class' : CLASS_NAMES[top_idx],
                'confidence'      : round(probs[top_idx], 4),
                'probabilities'   : {c: round(p,4) for c,p in zip(CLASS_NAMES, probs)},
            })
        except Exception as e:
            results.append({'filename': f.filename, 'error': str(e)})

    return jsonify({'predictions': results}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
