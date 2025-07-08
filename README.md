# Tree Species Classification with MobileNet Transfer Learning

This project implements a modular pipeline for classifying tree species from images using transfer learning with MobileNet architectures (V2, V3Large, V3Small) in TensorFlow/Keras.

## Project Structure

```
├── classify.py                # Main script to run the pipeline
├── requirements.txt           # Python dependencies
├── src/
│   ├── config.py              # Configuration (hyperparameters, paths, etc.)
│   ├── data_processor.py      # Data loading and preprocessing
│   ├── mobilenet_transfer_learning.py  # Model building, training, fine-tuning
│   ├── model_evaluator.py     # Evaluation and metrics
│   └── tflite_converter.py    # TFLite conversion utilities
└── data/
    ├── train/                 # Training images (one subfolder per class)
    ├── val/                   # Validation images
    └── test/                  # Test images
```

## Workflow Overview

1. **Configuration**
   - All settings (image size, batch size, learning rates, epochs, etc.) are defined in `src/config.py`.

2. **Data Processing**
   - `DataProcessor` (in `data_processor.py`) loads images from the `data/` directory using `tf.keras.utils.image_dataset_from_directory`.
   - Datasets are normalized, batched, and prefetched for efficient training.

3. **Model Building & Training**
   - `MobileNetTransferLearning` (in `mobilenet_transfer_learning.py`) builds a model using a pre-trained MobileNet backbone (V2, V3Large, or V3Small).
   - The base model is frozen and new classification layers are added.
   - The model is compiled and trained on the training set (feature extraction phase).
   - Callbacks (early stopping, learning rate reduction, model checkpointing) are used for robust training.

4. **Fine-Tuning**
   - After initial training, the base model is partially unfrozen (typically the last half of layers) and the model is re-trained with a lower learning rate.
   - This allows the model to adapt more specifically to the tree species dataset.

5. **Evaluation**
   - `ModelEvaluator` (in `model_evaluator.py`) provides tools for evaluating model performance on validation and test sets, including accuracy, confusion matrix, and classification reports.

6. **TFLite Conversion**
   - `TFLiteConverter` (in `tflite_converter.py`) enables conversion of trained models to TensorFlow Lite format for deployment on mobile or edge devices.

7. **Main Pipeline**
   - `classify.py` orchestrates the entire workflow: loading config, preparing data, training and fine-tuning the model, evaluating results, and exporting the model.

## How Transfer Learning Works in This Project

- **Transfer learning** leverages a MobileNet model pre-trained on ImageNet. The base model's weights are reused, and only the new classification layers are trained initially.
- **Feature extraction phase:** The base model is frozen, and only the new layers learn to classify tree species.
- **Fine-tuning phase:** Some layers of the base model are unfrozen, allowing the model to adapt more closely to the new dataset.
- This approach enables high accuracy even with limited data and reduces training time.

## Usage

1. **Install dependencies:**
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Prepare your data:**
   - Place images in `data/train/`, `data/val/`, and `data/test/` with one subfolder per class/species.
3. **Run the pipeline:**
   ```bash
   python classify.py
   ```
   - The script will train, fine-tune, evaluate, and (optionally) export the model.

## Customization
- Adjust hyperparameters and paths in `src/config.py`.
- Switch between MobileNet variants by changing the model name in the config or main script.

## Best Practices
- Use a Python 3.10 virtual environment for compatibility with TensorFlow.
- Keep `requirements.txt` updated for reproducibility.
- Modular code structure makes it easy to extend or swap components.

## References
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)

---

For questions or improvements, please open an issue or pull request.
