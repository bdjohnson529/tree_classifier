class EnhancedConfig:
    """
    Enhanced configuration demonstrating all the new architectural improvements.
    This configuration enables all advanced features for maximum performance.
    """
    
    # Models to train
    MODELS_TO_TRAIN = ["MobileNetV2"]

    # Data parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    TRAIN_SPLIT = 0.80
    VALIDATION_SPLIT = 0.25

    # Training parameters
    INITIAL_EPOCHS = 25
    FINE_TUNE_EPOCHS = 15
    INITIAL_LEARNING_RATE = 0.0005
    FINE_TUNE_LEARNING_RATE = 0.0001

    # Model parameters
    NUM_CLASSES = 23
    DROPOUT_RATE = 0.5
    NUM_FROZEN_LAYERS = 2

    # Enhanced Architecture Features - ALL ENABLED
    USE_ENHANCED_ARCHITECTURE = True      # Multi-branch architecture with attention
    USE_PROGRESSIVE_UNFREEZING = True     # Progressive unfreezing strategy
    USE_ADVANCED_AUGMENTATION = True      # Advanced data augmentation
    USE_ENSEMBLE_LEARNING = True          # Ensemble learning
    USE_TEST_TIME_AUGMENTATION = True     # Test-time augmentation

    # Progressive unfreezing parameters
    PROGRESSIVE_PHASES = 3

    # Ensemble learning parameters
    ENSEMBLE_STRATEGY = 'stacking'  # Use stacking for best performance
    ENSEMBLE_MODELS = ["MobileNetV2", "MobileNetV3Large", "MobileNetV3Small"]
    META_LEARNER_EPOCHS = 50

    # Test-time augmentation parameters
    TTA_AUGMENTATIONS = 7  # More augmentations for better accuracy

    # Advanced training parameters
    USE_MIXUP = True
    USE_CUTMIX = True
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING_FACTOR = 0.1

    # Learning rate scheduling
    USE_COSINE_ANNEALING = True
    USE_WARM_RESTARTS = True

    # Regularization
    USE_DROPOUT_SCHEDULING = True
    USE_WEIGHT_DECAY = True
    WEIGHT_DECAY_FACTOR = 1e-4

    # Paths
    DATA_DIR = "data"
    MODEL_SAVE_DIR = "saved_models"
    RESULTS_DIR = "results"


class ConservativeConfig:
    """
    Conservative configuration that enables only the most proven improvements.
    Good for users who want better performance without experimental features.
    """
    
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
    NUM_FROZEN_LAYERS = 2

    # Enhanced Architecture Features - CONSERVATIVE SELECTION
    USE_ENHANCED_ARCHITECTURE = False     # Keep original architecture
    USE_PROGRESSIVE_UNFREEZING = True     # Proven to work well
    USE_ADVANCED_AUGMENTATION = True      # Almost always beneficial
    USE_ENSEMBLE_LEARNING = False         # Keep it simple
    USE_TEST_TIME_AUGMENTATION = True     # Low risk, good reward

    # Progressive unfreezing parameters
    PROGRESSIVE_PHASES = 3

    # Ensemble learning parameters (not used)
    ENSEMBLE_STRATEGY = 'soft_voting'
    ENSEMBLE_MODELS = ["MobileNetV2"]
    META_LEARNER_EPOCHS = 50

    # Test-time augmentation parameters
    TTA_AUGMENTATIONS = 5

    # Advanced training parameters
    USE_MIXUP = False
    USE_CUTMIX = False
    USE_LABEL_SMOOTHING = True            # Usually beneficial
    LABEL_SMOOTHING_FACTOR = 0.1

    # Learning rate scheduling
    USE_COSINE_ANNEALING = False
    USE_WARM_RESTARTS = False

    # Regularization
    USE_DROPOUT_SCHEDULING = False
    USE_WEIGHT_DECAY = True               # Standard regularization
    WEIGHT_DECAY_FACTOR = 1e-4

    # Paths
    DATA_DIR = "data"
    MODEL_SAVE_DIR = "saved_models"
    RESULTS_DIR = "results"