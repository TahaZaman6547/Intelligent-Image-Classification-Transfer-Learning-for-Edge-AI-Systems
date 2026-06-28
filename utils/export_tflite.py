"""
Export the trained Keras model to TFLite format for Edge AI deployment.
TFLite models run on Raspberry Pi, Android, iOS, and microcontrollers.

Usage:
    python utils/export_tflite.py --model mobilenet_best.keras --output model.tflite
"""

import argparse
import numpy as np
import tensorflow as tf


def export_tflite(model_path: str, output_path: str, quantize: bool = False):
    """
    Convert a saved Keras model to TFLite.

    Args:
        model_path:  Path to the .keras / .h5 model file
        output_path: Output .tflite file path
        quantize:    Apply post-training dynamic range quantization
                     (reduces model size ~4× with minimal accuracy loss)
    """
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Applying dynamic range quantization...")

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {output_path}  ({size_mb:.2f} MB)")


def test_tflite(tflite_path: str):
    """Quick inference test on a random input."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    # Random 96×96 image
    test_input = np.random.rand(1, 96, 96, 3).astype(np.float32)
    # MobileNetV2 preprocessing
    test_input = (test_input * 2.0) - 1.0

    interpreter.set_tensor(in_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(out_details[0]['index'])

    CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
    pred_class = CLASS_NAMES[np.argmax(output[0])]
    print(f"Test inference OK → predicted class: {pred_class}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Keras model to TFLite')
    parser.add_argument('--model',    default='mobilenet_best.keras', help='Input .keras model path')
    parser.add_argument('--output',   default='model.tflite',          help='Output .tflite path')
    parser.add_argument('--quantize', action='store_true',              help='Apply quantization')
    args = parser.parse_args()

    export_tflite(args.model, args.output, args.quantize)
    test_tflite(args.output)
