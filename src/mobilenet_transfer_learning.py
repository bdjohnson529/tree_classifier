import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time

class MobileNetTransferLearning:
    """
    Transfer learning implementation for MobileNet models
    """

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.histories = {}

    def create_base_model(self, model_name):
        """
        Create the base MobileNet model based on the specified model name.
        """

        if model_name == "MobileNetV2":
            base_model = MobileNetV2(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        elif model_name == "MobileNetV3Large":
            base_model = MobileNetV3Large(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        elif model_name == "MobileNetV3Small":
            base_model = MobileNetV3Small(
                input_shape=(*self.config.IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return base_model

    def build_model(self, model_name):
        """
        Build the complete model with the base MobileNet and additional layers.
        Uses a multi-branch architecture with attention mechanism for better feature extraction.
        """

        base_model = self.create_base_model(model_name)

        # Unfreeze only the last N layers of the base model for feature extraction
        num_frozen = len(base_model.layers) - self.config.NUM_FROZEN_LAYERS
        for layer in base_model.layers[:num_frozen]:
            layer.trainable = False
        for layer in base_model.layers[num_frozen:]:
            layer.trainable = True

        # Check if enhanced architecture is enabled
        if hasattr(self.config, 'USE_ENHANCED_ARCHITECTURE') and self.config.USE_ENHANCED_ARCHITECTURE:
            return self._build_enhanced_model(base_model, model_name)
        else:
            # Original sequential model
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(self.config.DROPOUT_RATE),
                layers.Dense(self.config.NUM_CLASSES, activation='softmax')
            ])
            return model

    def _build_enhanced_model(self, base_model, model_name):
        """
        Build enhanced multi-branch architecture with attention mechanism.
        """
        inputs = base_model.input
        base_features = base_model(inputs)
        
        # Branch 1: Global Average Pooling (original approach)
        branch1 = layers.GlobalAveragePooling2D(name='global_avg_pool')(base_features)
        branch1 = layers.Dropout(self.config.DROPOUT_RATE)(branch1)
        branch1 = layers.Dense(256, activation='relu', name='branch1_dense')(branch1)
        
        # Branch 2: Global Max Pooling for complementary features
        branch2 = layers.GlobalMaxPooling2D(name='global_max_pool')(base_features)
        branch2 = layers.Dropout(self.config.DROPOUT_RATE)(branch2)
        branch2 = layers.Dense(256, activation='relu', name='branch2_dense')(branch2)
        
        # Branch 3: Attention mechanism
        attention_features = self._create_attention_branch(base_features)
        
        # Combine branches
        combined = layers.Concatenate(name='combine_branches')([branch1, branch2, attention_features])
        
        # Additional processing layers
        x = layers.Dense(512, activation='relu', name='combined_dense1')(combined)
        x = layers.Dropout(self.config.DROPOUT_RATE)(x)
        x = layers.Dense(256, activation='relu', name='combined_dense2')(x)
        x = layers.Dropout(self.config.DROPOUT_RATE / 2)(x)
        
        # Output layer
        outputs = layers.Dense(self.config.NUM_CLASSES, activation='softmax', name='predictions')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=f'{model_name}_enhanced')
        return model

    def _create_attention_branch(self, base_features):
        """
        Create attention mechanism branch for enhanced feature extraction.
        """
        # Spatial attention
        spatial_attention = layers.Conv2D(1, kernel_size=1, activation='sigmoid', name='spatial_attention')(base_features)
        attended_features = layers.Multiply(name='apply_spatial_attention')([base_features, spatial_attention])
        
        # Channel attention
        gap = layers.GlobalAveragePooling2D(name='attention_gap')(attended_features)
        channel_attention = layers.Dense(base_features.shape[-1] // 8, activation='relu', name='channel_attention_1')(gap)
        channel_attention = layers.Dense(base_features.shape[-1], activation='sigmoid', name='channel_attention_2')(channel_attention)
        channel_attention = layers.Reshape((1, 1, base_features.shape[-1]), name='reshape_channel_attention')(channel_attention)
        
        # Apply channel attention
        final_attention = layers.Multiply(name='apply_channel_attention')([attended_features, channel_attention])
        
        # Pool and process
        attention_pooled = layers.GlobalAveragePooling2D(name='attention_pool')(final_attention)
        attention_processed = layers.Dense(256, activation='relu', name='attention_dense')(attention_pooled)
        attention_processed = layers.Dropout(self.config.DROPOUT_RATE)(attention_processed)
        
        return attention_processed

    def compile_model(self, model, model_name, learning_rate=None):
        """
        Compile the model to prepare it for training.
        Uses the appropriate optimizer and loss function based on the model name.
        """

        if learning_rate is None:
            learning_rate = self.config.INITIAL_LEARNING_RATE
        if model_name == "MobileNetV2":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9
            )
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_feature_extraction(self, model_name, train_data, val_data):
        """
        Train the model in the feature extraction phase.
        This phase freezes the base model and trains only the top layers.
        """

        print(f"\n{'='*50}")
        print(f"Training {model_name} - Feature Extraction Phase")
        print(f"{'='*50}")

        model = self.build_model(model_name)
        model = self.compile_model(model, model_name)

        # Callbacks are used to monitor training and adjust learning rate or stop early
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=3),
            ModelCheckpoint(
                f"{self.config.MODEL_SAVE_DIR}/{model_name}_feature_extraction.h5",
                save_best_only=True
            )
        ]

        # Start training and measure time
        start_time = time.time()
        history = model.fit(
            train_data,
            epochs=self.config.INITIAL_EPOCHS,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model parameters: {model.count_params():,}")

        self.models[f"{model_name}_feature_extraction"] = model
        self.histories[f"{model_name}_feature_extraction"] = history

        return model, history, training_time

    def fine_tune_model(self, model, model_name, train_data, val_data):
        """
        Fine-tune the model by unfreezing some layers of the base model.
        This phase allows the model to learn more specific features.
        """

        print(f"\n{'='*50}")
        print(f"Fine-tuning {model_name}")
        print(f"{'='*50}")

        # Check if progressive unfreezing is enabled
        if hasattr(self.config, 'USE_PROGRESSIVE_UNFREEZING') and self.config.USE_PROGRESSIVE_UNFREEZING:
            return self._progressive_fine_tune(model, model_name, train_data, val_data)
        else:
            return self._standard_fine_tune(model, model_name, train_data, val_data)

    def _standard_fine_tune(self, model, model_name, train_data, val_data):
        """Standard fine-tuning approach (original method)."""
        # Unfreeze the base model and set the first half of its layers to be trainable
        base_model = model.layers[0] if hasattr(model, 'layers') else model.get_layer(index=0)
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) // 2
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        model = self.compile_model(
            model, 
            model_name, 
            learning_rate=self.config.FINE_TUNE_LEARNING_RATE
        )

        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5),
            ModelCheckpoint(
                f"{self.config.MODEL_SAVE_DIR}/{model_name}_fine_tuned.h5",
                save_best_only=True
            )
        ]

        start_time = time.time()
        history = model.fit(
            train_data,
            epochs=self.config.FINE_TUNE_EPOCHS,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )

        fine_tune_time = time.time() - start_time
        print(f"Fine-tuning time: {fine_tune_time:.2f} seconds")
        self.models[f"{model_name}_fine_tuned"] = model
        self.histories[f"{model_name}_fine_tuned"] = history

        return model, history, fine_tune_time

    def _progressive_fine_tune(self, model, model_name, train_data, val_data):
        """Progressive unfreezing with multiple phases."""
        print("Using progressive unfreezing strategy...")
        
        # Find base model layer
        base_model = None
        for layer in model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 20:  # Likely the base model
                base_model = layer
                break
        
        if base_model is None:
            print("Warning: Could not find base model for progressive unfreezing. Using standard method.")
            return self._standard_fine_tune(model, model_name, train_data, val_data)

        total_layers = len(base_model.layers)
        phases = getattr(self.config, 'PROGRESSIVE_PHASES', 3)
        epochs_per_phase = self.config.FINE_TUNE_EPOCHS // phases
        
        all_histories = []
        total_time = 0
        
        for phase in range(phases):
            print(f"\n--- Progressive Fine-tuning Phase {phase + 1}/{phases} ---")
            
            # Calculate which layers to unfreeze for this phase
            unfreeze_from = total_layers - ((phase + 1) * total_layers // phases)
            
            # Set trainability
            for i, layer in enumerate(base_model.layers):
                layer.trainable = i >= unfreeze_from
            
            # Adjust learning rate for each phase
            phase_lr = self.config.FINE_TUNE_LEARNING_RATE * (0.5 ** phase)
            model = self.compile_model(model, model_name, learning_rate=phase_lr)
            
            print(f"Unfreezing layers from index {unfreeze_from} onwards ({total_layers - unfreeze_from} layers)")
            print(f"Learning rate: {phase_lr:.6f}")
            
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=3),
                ModelCheckpoint(
                    f"{self.config.MODEL_SAVE_DIR}/{model_name}_progressive_phase_{phase+1}.h5",
                    save_best_only=True
                )
            ]
            
            start_time = time.time()
            history = model.fit(
                train_data,
                epochs=epochs_per_phase,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=1
            )
            
            phase_time = time.time() - start_time
            total_time += phase_time
            all_histories.append(history)
            print(f"Phase {phase + 1} time: {phase_time:.2f} seconds")
        
        # Combine all histories
        combined_history = self._combine_histories(all_histories)
        
        # Save final model
        model.save(f"{self.config.MODEL_SAVE_DIR}/{model_name}_fine_tuned.h5")
        
        print(f"Total fine-tuning time: {total_time:.2f} seconds")
        self.models[f"{model_name}_fine_tuned"] = model
        self.histories[f"{model_name}_fine_tuned"] = combined_history
        
        return model, combined_history, total_time

    def _combine_histories(self, histories):
        """Combine multiple training histories into one."""
        if not histories:
            return None
        
        combined = {}
        for key in histories[0].history.keys():
            combined[key] = []
            for history in histories:
                combined[key].extend(history.history[key])
        
        class CombinedHistory:
            def __init__(self, history_dict):
                self.history = history_dict
        
        return CombinedHistory(combined)
