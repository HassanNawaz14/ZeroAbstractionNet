"""Single source of truth for all hyperparameters and dataset defaults."""

# Network architecture
LAYER_SIZES = [2, 4, 4, 1]

# Training (defaults = demo tier per the two-tier strategy in README.md;
# showcase tier is always explicit CLI flags)
LR = 2.5
SEED = 0
N_PER_QUADRANT = 25
LOG_EVERY = 5
EPOCHS = 250

# Dataset
NOISE_STD = 0.0
PROBE_RESOLUTION = 40
