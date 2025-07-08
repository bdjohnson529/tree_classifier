"""
MobileNetV3 Transfer Learning for Tree Species Identification
Replication of the study: "Performance of MobileNetV3 Transfer Learning on
Handheld Device-based Real-Time Tree Species Identification"

This code implements the methodology described in the paper, comparing
MobileNetV2, MobileNetV3-Large, and MobileNetV3-Small for leaf classification.
"""


import os
import tensorflow as tf
from src.config import Config
from src.data_processor import DataProcessor
from src.mobilenet_transfer_learning import MobileNetTransferLearning
from src.model_evaluator import ModelEvaluator
from src.tflite_converter import TensorFlowLiteConverter


def main():
    """Main function to run the complete pipeline"""
    
    # Initialize configuration
    config = Config()
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    print("MobileNetV3 Transfer Learning for Tree Species Identification")
    print("="*60)
    
    # Initialize components
    data_processor = DataProcessor(config)
    transfer_learning = MobileNetTransferLearning(config)
    evaluator = ModelEvaluator(config)
    tflite_converter = TensorFlowLiteConverter(config)
    
    # Load and prepare data
    print("\n1. Loading and preparing dataset...")
    
    # Option 1: If you have separate train/validation directories
    train_data, val_data = data_processor.create_data_generators(
        train_dir="data/train",
        validation_dir="data/val"
    )    
    
    # Models to compare (as in the paper)
    models_to_train = ["MobileNetV3Small"]
    
    all_results = {}
    training_times = {}
    
    # Train each model
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name}")
        print(f"{'='*60}")
        
        # Feature extraction phase
        model, history, train_time = transfer_learning.train_feature_extraction(
            model_name, train_data, val_data
        )
        
        # Fine-tuning phase
        model, fine_tune_history, fine_tune_time = transfer_learning.fine_tune_model(
            model, model_name, train_data, val_data
        )
        
        # Evaluate model
        predicted_classes, true_classes = evaluator.evaluate_model(
            model, val_data, model_name
        )
        
        # Convert to TensorFlow Lite
        tflite_path, tflite_size = tflite_converter.convert_to_tflite(model, model_name)
        
        # Store timing information
        training_times[model_name] = {
            'feature_extraction_time': train_time,
            'fine_tuning_time': fine_tune_time,
            'total_time': train_time + fine_tune_time
        }
    
    # Generate final comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    # Plot training histories
    evaluator.plot_training_history(transfer_learning.histories)
    
    # Create comparison table
    comparison_df = evaluator.create_comparison_table()
    
    # Print training times
    print(f"\n{'='*60}")
    print("TRAINING TIMES COMPARISON")
    print(f"{'='*60}")
    for model_name, times in training_times.items():
        print(f"{model_name}:")
        print(f"  Feature Extraction: {times['feature_extraction_time']:.2f}s")
        print(f"  Fine-tuning: {times['fine_tuning_time']:.2f}s")
        print(f"  Total: {times['total_time']:.2f}s")
    
    print(f"\nAll models and results saved in:")
    print(f"- Models: {config.MODEL_SAVE_DIR}/")
    print(f"- Results: {config.RESULTS_DIR}/")

if __name__ == "__main__":
    # Check TensorFlow GPU availability
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    
    # Run main pipeline
    main()