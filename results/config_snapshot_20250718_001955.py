class Config:
    # Models to train
    MODELS_TO_TRAIN = ["MobileNetV2"]

    # Data parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.80
    VALIDATION_SPLIT = 0.25

    # Training parameters
    INITIAL_EPOCHS = 20
    FINE_TUNE_EPOCHS = 10
    INITIAL_LEARNING_RATE = 0.0005
    FINE_TUNE_LEARNING_RATE = 0.0001

    # Model parameters
    NUM_CLASSES = 23
    DROPOUT_RATE = 0.5
    NUM_FROZEN_LAYERS = 2  # Number of layers to freeze in feature extraction

    # Enhanced Architecture Features (set to True to enable)
    USE_ENHANCED_ARCHITECTURE = False  # Multi-branch architecture with attention
    USE_PROGRESSIVE_UNFREEZING = False  # Progressive unfreezing strategy
    USE_ADVANCED_AUGMENTATION = False  # Advanced data augmentation
    USE_ENSEMBLE_LEARNING = False  # Ensemble learning
    USE_TEST_TIME_AUGMENTATION = False  # Test-time augmentation

    # Progressive unfreezing parameters
    PROGRESSIVE_PHASES = 3  # Number of progressive unfreezing phases
    
    # Ensemble learning parameters
    ENSEMBLE_STRATEGY = 'soft_voting'  # 'soft_voting', 'hard_voting', 'weighted_voting', 'stacking'
    ENSEMBLE_MODELS = ["MobileNetV2", "MobileNetV3Large", "MobileNetV3Small"]
    META_LEARNER_EPOCHS = 50
    
    # Test-time augmentation parameters
    TTA_AUGMENTATIONS = 5  # Number of augmentations per test image
    
    # Advanced training parameters
    USE_MIXUP = False  # Mixup data augmentation
    USE_CUTMIX = False  # CutMix data augmentation
    USE_LABEL_SMOOTHING = False  # Label smoothing
    LABEL_SMOOTHING_FACTOR = 0.1
    
    # Learning rate scheduling
    USE_COSINE_ANNEALING = False  # Cosine annealing scheduler
    USE_WARM_RESTARTS = False  # Warm restarts
    
    # Regularization
    USE_DROPOUT_SCHEDULING = False  # Adaptive dropout scheduling
    USE_WEIGHT_DECAY = False  # L2 weight decay
    WEIGHT_DECAY_FACTOR = 1e-4

    # Paths
    DATA_DIR = "data"
    MODEL_SAVE_DIR = "saved_models"
    RESULTS_DIR = "results"
