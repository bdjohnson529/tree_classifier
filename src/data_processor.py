import tensorflow as tf
from tensorflow.keras import layers

class DataProcessor:
    """Handle dataset loading and preprocessing"""
    def __init__(self, config):
        self.config = config

    def create_data_generators(self, train_dir, validation_dir=None):
        """
        Create train and validation datasets using tf.keras.utils.image_dataset_from_directory.
        If validation_dir is not provided, split train_dir using validation_split.
        """

        # Choose augmentation strategy based on config
        if hasattr(self.config, 'USE_ADVANCED_AUGMENTATION') and self.config.USE_ADVANCED_AUGMENTATION:
            data_augmentation = self._create_advanced_augmentation()
        else:
            data_augmentation = self._create_basic_augmentation()

        if validation_dir:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                image_size=self.config.IMG_SIZE,
                batch_size=self.config.BATCH_SIZE,
                label_mode='categorical',
                shuffle=True
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                validation_dir,
                image_size=self.config.IMG_SIZE,
                batch_size=self.config.BATCH_SIZE,
                label_mode='categorical',
                shuffle=False
            )
        else:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                validation_split=self.config.VALIDATION_SPLIT,
                subset="training",
                seed=123,
                image_size=self.config.IMG_SIZE,
                batch_size=self.config.BATCH_SIZE,
                label_mode='categorical',
                shuffle=True
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                train_dir,
                validation_split=self.config.VALIDATION_SPLIT,
                subset="validation",
                seed=123,
                image_size=self.config.IMG_SIZE,
                batch_size=self.config.BATCH_SIZE,
                label_mode='categorical',
                shuffle=False
            )

        normalization_layer = tf.keras.layers.Rescaling(1./255)

        # Apply data augmentation only to training data
        train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        AUTOTUNE = tf.data.AUTOTUNE

        # Cache and prefetch datasets for performance
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # Debug: Print a batch of images and labels from the training set
        for images, labels in train_ds.take(1):
            print("Images shape:", images.shape)
            print("Labels:", labels.numpy())

        return train_ds, val_ds

    def _create_basic_augmentation(self):
        """Create basic data augmentation pipeline."""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

    def _create_advanced_augmentation(self):
        """Create advanced data augmentation pipeline with more sophisticated techniques."""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomTranslation(0.1, 0.1),
            # Custom augmentation for better leaf/tree feature learning
            layers.Lambda(lambda x: tf.image.random_hue(x, 0.1)),
            layers.Lambda(lambda x: tf.image.random_saturation(x, 0.8, 1.2)),
            layers.Lambda(lambda x: tf.image.random_jpeg_quality(x, 85, 100)),
        ])

    def create_test_time_augmentation_dataset(self, test_ds, n_augmentations=5):
        """
        Create test-time augmentation dataset for improved inference accuracy.
        """
        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomBrightness(0.1),
        ])
        
        def augment_batch(x, y):
            # Create multiple augmented versions of each image
            augmented_images = []
            for _ in range(n_augmentations):
                aug_x = augmentation(x)
                augmented_images.append(aug_x)
            
            # Stack all augmented versions
            stacked = tf.stack(augmented_images, axis=1)  # Shape: (batch, n_aug, height, width, channels)
            
            # Reshape to treat each augmentation as a separate sample
            batch_size = tf.shape(x)[0]
            reshaped = tf.reshape(stacked, (-1, *x.shape[1:]))
            
            # Repeat labels for each augmentation
            repeated_labels = tf.repeat(y, n_augmentations, axis=0)
            
            return reshaped, repeated_labels
        
        tta_ds = test_ds.map(augment_batch)
        return tta_ds, n_augmentations

    def prepare_dataset_from_directory(self, data_dir):
        """
        Prepare dataset from a directory structure with subdirectories for each class.
        Uses validation_split to create training and validation datasets.
        """

        # Data augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.config.VALIDATION_SPLIT,
            subset="training",
            seed=123,
            image_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.config.VALIDATION_SPLIT,
            subset="validation",
            seed=123,
            image_size=self.config.IMG_SIZE,
            batch_size=self.config.BATCH_SIZE
        )

        # Apply normalization and data augmentation
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

        AUTOTUNE = tf.data.AUTOTUNE

        # Cache and prefetch datasets for performance
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        return train_ds, val_ds
