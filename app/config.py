import os

# ======================
# System
# ======================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================
# Model
# ======================
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 5
DROPOUT = 0.1
CONTEXT_MODE = "relaxed"

# ======================
# Training
# ======================
LR = 5e-4
BATCH_SIZE = 256
EPOCHS = 25
PATIENCE = 5
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
LOSS_STAGE = 4
LOSS_SCHEDULE = "epoch"
STAGE_SIZE = 5
TRAIN_MONITOR = "ret_mae_skill"
TRAIN_MONITOR_MIN_IMPROVEMENT = 0.0
SAVE_BEST_CHECKPOINT = True
SEED = 42
DETERMINISTIC = False

# ======================
# Runtime
# ======================
DEFAULT_DEVICE = "cpu"   # "auto" | "cpu" | "cuda"
