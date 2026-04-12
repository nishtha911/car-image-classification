import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "training_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "testing_images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "train_solution_bounding_boxes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Hyperparameters
IMG_SIZE = (320, 320)  # Input size for EfficientNet
BATCH_SIZE = 8
EPOCHS = 20
NUM_CLASSES = 1  # Only 'car'
CLASS_NAMES = ["car"]

# Optimization Comparison
OPTIMIZERS = ["adam", "sgd", "rmsprop", "adamw"]
LEARNING_RATE = 1e-4
