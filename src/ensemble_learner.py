import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import List, Dict, Tuple

class EnsembleLearner:
    """
    Ensemble learning implementation for combining multiple models.
    Supports various ensemble strategies including voting, averaging, and stacking.
    """
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.model_weights = []
        self.ensemble_strategy = getattr(config, 'ENSEMBLE_STRATEGY', 'soft_voting')
        
    def add_model(self, model, weight=1.0):
        """Add a trained model to the ensemble."""
        self.models.append(model)
        self.model_weights.append(weight)
        
    def create_stacked_ensemble(self, models_dict: Dict, train_data, val_data):
        """
        Create a stacked ensemble using a meta-learner.
        The meta-learner learns to combine predictions from base models.
        """
        print("Creating stacked ensemble...")
        
        # Generate meta-features from base models
        meta_train_features, meta_train_labels = self._generate_meta_features(
            models_dict, train_data
        )
        meta_val_features, meta_val_labels = self._generate_meta_features(
            models_dict, val_data
        )
        
        # Create meta-learner
        meta_learner = self._create_meta_learner(len(models_dict))
        
        # Train meta-learner
        meta_learner.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = meta_learner.fit(
            meta_train_features, meta_train_labels,
            validation_data=(meta_val_features, meta_val_labels),
            epochs=getattr(self.config, 'META_LEARNER_EPOCHS', 50),
            batch_size=32,
            verbose=1
        )
        
        # Create final ensemble model
        ensemble_model = self._create_ensemble_model(models_dict, meta_learner)
        
        return ensemble_model, history
    
    def _generate_meta_features(self, models_dict: Dict, dataset):
        """Generate meta-features by collecting predictions from all base models."""
        all_predictions = []
        labels = []
        
        # Collect predictions from each model
        for model_name, model in models_dict.items():
            predictions = model.predict(dataset, verbose=0)
            all_predictions.append(predictions)
            
        # Collect true labels (only once)
        for _, batch_labels in dataset:
            labels.extend(batch_labels.numpy())
            
        # Stack predictions horizontally
        meta_features = np.hstack(all_predictions)
        meta_labels = np.array(labels[:len(meta_features)])
        
        return meta_features, meta_labels
    
    def _create_meta_learner(self, num_base_models):
        """Create the meta-learner neural network."""
        input_dim = num_base_models * self.config.NUM_CLASSES
        
        meta_learner = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        return meta_learner
    
    def _create_ensemble_model(self, base_models: Dict, meta_learner):
        """Create the final ensemble model that combines base models with meta-learner."""
        
        # Create inputs for the ensemble
        input_layer = layers.Input(shape=(*self.config.IMG_SIZE, 3))
        
        # Get predictions from all base models
        base_predictions = []
        for model_name, model in base_models.items():
            # Make base model non-trainable
            model.trainable = False
            pred = model(input_layer)
            base_predictions.append(pred)
        
        # Concatenate all base predictions
        combined_predictions = layers.Concatenate()(base_predictions)
        
        # Pass through meta-learner
        final_prediction = meta_learner(combined_predictions)
        
        # Create final ensemble model
        ensemble_model = models.Model(inputs=input_layer, outputs=final_prediction)
        
        return ensemble_model
    
    def predict_with_ensemble(self, models_dict: Dict, test_data, strategy='soft_voting'):
        """
        Make predictions using ensemble strategy.
        
        Args:
            models_dict: Dictionary of trained models
            test_data: Test dataset
            strategy: 'soft_voting', 'hard_voting', 'weighted_voting'
        """
        print(f"Making ensemble predictions with strategy: {strategy}")
        
        # Collect predictions from all models
        all_predictions = []
        for model_name, model in models_dict.items():
            predictions = model.predict(test_data, verbose=0)
            all_predictions.append(predictions)
        
        # Apply ensemble strategy
        if strategy == 'soft_voting':
            # Average the probabilities
            ensemble_pred = np.mean(all_predictions, axis=0)
        elif strategy == 'hard_voting':
            # Majority voting on predicted classes
            hard_preds = [np.argmax(pred, axis=1) for pred in all_predictions]
            ensemble_pred = np.array([
                np.bincount(votes).argmax() 
                for votes in zip(*hard_preds)
            ])
        elif strategy == 'weighted_voting':
            # Weighted average based on model performance
            weights = np.array(self.model_weights)
            weights = weights / weights.sum()  # Normalize weights
            ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")
        
        return ensemble_pred
    
    def evaluate_ensemble_diversity(self, models_dict: Dict, test_data):
        """
        Evaluate the diversity of models in the ensemble.
        Higher diversity often leads to better ensemble performance.
        """
        print("Evaluating ensemble diversity...")
        
        predictions = {}
        for model_name, model in models_dict.items():
            pred = model.predict(test_data, verbose=0)
            predictions[model_name] = np.argmax(pred, axis=1)
        
        # Calculate pairwise disagreement
        model_names = list(predictions.keys())
        disagreements = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                disagreement = np.mean(predictions[model1] != predictions[model2])
                disagreements[f"{model1}_vs_{model2}"] = disagreement
        
        # Calculate average disagreement
        avg_disagreement = np.mean(list(disagreements.values()))
        
        print(f"Average pairwise disagreement: {avg_disagreement:.4f}")
        print("Pairwise disagreements:")
        for pair, disagreement in disagreements.items():
            print(f"  {pair}: {disagreement:.4f}")
        
        return disagreements, avg_disagreement
    
    def create_model_ensemble_config(self, base_models: List[str]):
        """
        Create configuration for training multiple models for ensemble.
        """
        ensemble_configs = []
        
        for i, model_name in enumerate(base_models):
            config = {
                'model_name': model_name,
                'seed': 42 + i,  # Different seeds for diversity
                'dropout_rate': self.config.DROPOUT_RATE + (i * 0.1),  # Varying dropout
                'data_augmentation_strength': 1.0 + (i * 0.2),  # Varying augmentation
                'initial_lr': self.config.INITIAL_LEARNING_RATE * (0.8 ** i),  # Varying LR
            }
            ensemble_configs.append(config)
        
        return ensemble_configs