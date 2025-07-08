import os
import tensorflow as tf

class TensorFlowLiteConverter:
    """Convert models to TensorFlow Lite for mobile deployment"""
    def __init__(self, config):
        self.config = config

    def convert_to_tflite(self, model, model_name):
        print(f"\nConverting {model_name} to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = f"{self.config.MODEL_SAVE_DIR}/{model_name}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"TensorFlow Lite model saved: {tflite_path}")
        print(f"TensorFlow Lite model size: {size_mb:.2f} MB")
        return tflite_path, size_mb
