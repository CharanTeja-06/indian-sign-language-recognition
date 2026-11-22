"""
Central Configuration — ISL Word-to-Text Recognition
All paths, hyperparameters, and thresholds in one place.
"""

from pathlib import Path
import torch

# ==============================================================================
# 1. Target Classes (5 ISL words — NO "COLLEGE")
# ==============================================================================
TARGET_CLASSES = ["YOU", "WORK", "DOCTOR", "PAIN", "AGREE"]
NUM_CLASSES = len(TARGET_CLASSES)

# ==============================================================================
# 2. Paths
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset: class folders live in ./dataset or root directory
_dataset_candidate = PROJECT_ROOT / "dataset"
DATA_DIR = _dataset_candidate if _dataset_candidate.is_dir() else PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

# ==============================================================================
# 3. Image Preprocessing
# ==============================================================================
IMAGE_SIZE = (224, 224)

# EfficientNet-B0 / ImageNet normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# ==============================================================================
# 4. Dataset Splitting
# ==============================================================================
VAL_RATIO = 0.20          # 80% train, 20% val — stratified, all users in both
RANDOM_SEED = 42

# ==============================================================================
# 5. Model & Training
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture: EfficientNet-B0 (best accuracy within 4GB VRAM constraint)
MODEL_NAME = "efficientnet_b0"

BATCH_SIZE = 32           # fits in 4GB VRAM with AMP; fall back to 16 if OOM
NUM_WORKERS = 2           # 2 workers for Windows with GPU

DROPOUT_RATE = 0.3

# Stage 1: Frozen backbone — train classifier head
STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3

# Stage 2: Fine-tune all layers
STAGE2_EPOCHS = 25
STAGE2_LR = 3e-4

WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8

# Training loss uses mild smoothing for regularization
TRAIN_LABEL_SMOOTHING = 0.05
# Validation loss uses NO smoothing — true cross-entropy
VAL_LABEL_SMOOTHING = 0.0

# ==============================================================================
# 6. Webcam Application
# ==============================================================================
WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES_REQUIRED = 8
MP_MIN_DETECTION_CONFIDENCE = 0.5
