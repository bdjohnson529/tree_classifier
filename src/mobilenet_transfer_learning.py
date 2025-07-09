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
        Uses a sequential model which means the input is a single tensor.
        Global average pooling is applied to reduce the spatial dimensions,
        followed by a dropout layer and a dense output layer with softmax activation.
        """

        base_model = self.create_base_model(model_name)

        # Unfreeze the last 20 layers of the base model for feature extraction
        for layer in base_model.layers[:-1]:
            layer.trainable = False
        for layer in base_model.layers[-1:]:
            layer.trainable = True

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])

        return model

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

        # Unfreeze the base model and set the first half of its layers to be trainable
        base_model = model.layers[0]
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
