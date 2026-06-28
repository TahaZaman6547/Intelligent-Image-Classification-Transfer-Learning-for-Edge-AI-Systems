"""
Standalone inference script — classify a local image using the trained model.

Usage:
    python utils/predict_image.py --image path/to/image.jpg --model mobilenet_best.keras
"""

import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']


def predict(image_path: str, model_path: str, img_size: int = 96):
    model = tf.keras.models.load_model(model_path)

    img = Image.open(image_path).convert('RGB').resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    top5  = np.argsort(probs)[::-1][:5]

    print(f"\nImage      : {image_path}")
    print(f"Prediction : {CLASS_NAMES[top5[0]]}  ({probs[top5[0]]*100:.2f}%)")
    print("\nTop-5 predictions:")
    for i, idx in enumerate(top5):
        print(f"  {i+1}. {CLASS_NAMES[idx]:<12} {probs[idx]*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--model', default='mobilenet_best.keras')
    parser.add_argument('--size',  type=int, default=96)
    args = parser.parse_args()
    predict(args.image, args.model, args.size)
