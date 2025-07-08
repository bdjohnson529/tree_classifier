import os
import pandas as pd
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Evaluate and compare model performance"""
    def __init__(self, config):
        self.config = config
        self.results = {}

    def evaluate_model(self, model, test_data, model_name):
        """
        Evaluate the model on the test dataset and return predictions and true labels.
        Also calculates model size and number of parameters.
        """

        print(f"\nEvaluating {model_name}...")
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

    def get_model_size(self, model):
        temp_path = "temp_model.h5"
        model.save(temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb

    def plot_training_history(self, histories):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Training Comparison', fontsize=16)
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
        import datetime
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{self.config.RESULTS_DIR}/training_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_comparison_table(self):
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
        df.to_csv(f'{self.config.RESULTS_DIR}/model_comparison.csv')
        return df
