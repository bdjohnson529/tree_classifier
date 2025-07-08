class Config:
    # Data parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    TRAIN_SPLIT = 0.75
    VALIDATION_SPLIT = 0.25

    # Training parameters
    INITIAL_EPOCHS = 2
    FINE_TUNE_EPOCHS = 1
    INITIAL_LEARNING_RATE = 0.003
    FINE_TUNE_LEARNING_RATE = 0.0001

    # Model parameters
    NUM_CLASSES = 23
    DROPOUT_RATE = 0.4

    # Paths
    DATA_DIR = "data"
    MODEL_SAVE_DIR = "saved_models"
    RESULTS_DIR = "results"
