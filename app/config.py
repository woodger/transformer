# ======================
# Model
# ======================
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 2
DROPOUT = 0.1

# ======================
# Training
# ======================
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# ======================
# Data
# ======================
SEQ_LEN = 1

# ======================
# Runtime
# ======================
DEFAULT_DEVICE = "cpu"   # "cpu" | "gpu"