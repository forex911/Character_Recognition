from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "model"

MODEL_PATH = MODEL_DIR / "model.pth"

# Training
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3
NUM_CLASSES = 26

# Image
IMG_SIZE = 28
