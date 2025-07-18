import os
import datetime
import pandas as pd
import shutil
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Evaluate and compare model performance"""
    def __init__(self, config):
        self.config = config
        self.results = {}
        self._shared_timestamp = None  # For consistent file naming

    def _get_or_create_timestamp(self):
        if self._shared_timestamp is None:
            self._shared_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return self._shared_timestamp

    def evaluate_model(self, model, test_data, model_name):
        """
        Evaluate the model on the test dataset and return predictions and true labels.
        Also calculates model size and number of parameters.
        """

        print(f"\nEvaluating {model_name}...")
        
        # Check if test-time augmentation is enabled
        if hasattr(self.config, 'USE_TEST_TIME_AUGMENTATION') and self.config.USE_TEST_TIME_AUGMENTATION:
            predictions = self._evaluate_with_tta(model, test_data, model_name)
        else:
            predictions = model.predict(test_data)

        import numpy as np
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = []

        for _, labels in test_data:
            true_classes.extend(np.argmax(labels.numpy(), axis=1))
        true_classes = np.array(true_classes[:len(predicted_classes)])
        test_loss, test_accuracy = model.evaluate(test_data, verbose=0)

        self.results[model_name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'model_size_mb': self.get_model_size(model),
            'parameters': model.count_params()
        }

        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Model Size: {self.results[model_name]['model_size_mb']:.2f} MB")

        return predicted_classes, true_classes

    def _evaluate_with_tta(self, model, test_data, model_name):
        """
        Evaluate model with test-time augmentation for improved accuracy.
        """
        from .data_processor import DataProcessor
        
        print(f"Using test-time augmentation for {model_name}...")
        
        # Create data processor for TTA
        data_processor = DataProcessor(self.config)
        
        # Create TTA dataset
        tta_dataset, n_augmentations = data_processor.create_test_time_augmentation_dataset(
            test_data, self.config.TTA_AUGMENTATIONS
        )
        
        # Get predictions on augmented data
        tta_predictions = model.predict(tta_dataset, verbose=0)
        
        # Reshape predictions to group by original image
        n_original_samples = len(tta_predictions) // n_augmentations
        reshaped_predictions = tta_predictions.reshape(n_original_samples, n_augmentations, -1)
        
        # Average predictions across augmentations
        final_predictions = np.mean(reshaped_predictions, axis=1)
        
        print(f"TTA completed with {n_augmentations} augmentations per image")
        
        return final_predictions

    def get_model_size(self, model):
        """
        Calculate the size of the model in MB.
        """

        temp_path = "temp_model.h5"
        model.save(temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb

    def create_comparison_table(self, timestamp=None):
        """
        Create a comparison table of model results and save with a timestamp.
        """
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df = df.round(4)

        column_order = ['test_accuracy', 'test_loss', 'model_size_mb', 'parameters']
        df = df[column_order]
        df.columns = ['Accuracy (%)', 'Loss', 'Size (MB)', 'Parameters']
        df['Accuracy (%)'] = df['Accuracy (%)'] * 100

        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(df.to_string())

        # Save with and without timestamp
        if timestamp is not None:
            df.to_csv(f'{self.config.RESULTS_DIR}/model_comparison_{timestamp}.csv')

        return df

    def plot_training_history(self, histories):
        """
        Plot training and validation accuracy/loss for all models.
        Saves the plots and training history to CSV.
        Calls create_comparison_table to save model comparison with the same timestamp.
        """

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training Comparison', fontsize=16)

        # Prepare to save all histories to CSV
        all_history = []
        for i, (model_name, history) in enumerate(histories.items()):
            row = i // 2
            col = i % 2
            if i < 4:
                axes[row, col].plot(history.history['accuracy'], label='Training Accuracy')
                axes[row, col].plot(history.history['val_accuracy'], label='Validation Accuracy')
                axes[row, col].set_title(f'{model_name} - Accuracy')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].legend()
                axes[row, col].grid(True)

            # Save history to all_history for CSV
            for epoch in range(len(history.history['accuracy'])):
                all_history.append({
                    'model': model_name,
                    'epoch': epoch + 1,
                    'train_accuracy': history.history['accuracy'][epoch],
                    'val_accuracy': history.history['val_accuracy'][epoch],
                    'train_loss': history.history['loss'][epoch],
                    'val_loss': history.history['val_loss'][epoch]
                })

        plt.tight_layout()
        timestamp = self._get_or_create_timestamp()
        plt.savefig(f'{self.config.RESULTS_DIR}/training_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save all histories to CSV
        df = pd.DataFrame(all_history)
        df.to_csv(f'{self.config.RESULTS_DIR}/training_history_{timestamp}.csv', index=False)

        # Save a snapshot of config.py with the same timestamp
        config_src_path = os.path.join(os.path.dirname(__file__), 'config.py')
        config_dst_path = os.path.join(self.config.RESULTS_DIR, f'config_snapshot_{timestamp}.py')
        try:
            shutil.copyfile(config_src_path, config_dst_path)
        except Exception as e:
            print(f"Warning: Could not save config snapshot: {e}")

        # Call create_comparison_table with the same timestamp
        self.create_comparison_table(timestamp=timestamp)
