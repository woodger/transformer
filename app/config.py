import os

# ======================
# System
# ======================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# Model
# ======================
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 4
DROPOUT = 0.1

# ======================
# Training
# ======================
LR = 5e-4
BATCH_SIZE = 512
EPOCHS = 50
PATIENCE = 5
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0

# ======================
# Runtime
# ======================
DEFAULT_DEVICE = "cpu"   # "cpu" | "gpu"