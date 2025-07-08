import os
import datetime
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
        """
        Calculate the size of the model in MB.
        """

        temp_path = "temp_model.h5"
        model.save(temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size_mb

    def plot_training_history(self, histories):
        """
        Plot training and validation accuracy/loss for all models.
        Saves the plots and training history to CSV.
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
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{self.config.RESULTS_DIR}/training_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Save all histories to CSV
        df = pd.DataFrame(all_history)
        df.to_csv(f'{self.config.RESULTS_DIR}/training_history_{timestamp}.csv', index=False)

    def create_comparison_table(self):
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
        df.to_csv(f'{self.config.RESULTS_DIR}/model_comparison.csv')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        df.to_csv(f'{self.config.RESULTS_DIR}/model_comparison_{timestamp}.csv')

        return df
