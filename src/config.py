class Config:
    # Models to train
    MODELS_TO_TRAIN = ["MobileNetV2"]

    # Data parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.80
    VALIDATION_SPLIT = 0.25

    # Training parameters
    INITIAL_EPOCHS = 15
    FINE_TUNE_EPOCHS = 10
    INITIAL_LEARNING_RATE = 0.001
    FINE_TUNE_LEARNING_RATE = 0.0001

    # Model parameters
    NUM_CLASSES = 23
    DROPOUT_RATE = 0.3

    # Paths
    DATA_DIR = "data"
    MODEL_SAVE_DIR = "saved_models"
    RESULTS_DIR = "results"
