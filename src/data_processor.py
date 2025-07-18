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

        # Data augmentation pipeline
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])

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
