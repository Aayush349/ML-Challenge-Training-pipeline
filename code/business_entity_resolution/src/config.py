import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Dataset directories with dynamic fallback detection
DATASET_CANDIDATES = [
    Path.cwd() / "dataset",
    Path.cwd() / "student_resource" / "dataset",
    Path("/content/student_resource/dataset"),
    Path("/content/student_resource/student_resource/dataset"),
    PROJECT_ROOT / "dataset",
    PROJECT_ROOT / "student_resource" / "dataset",
]

DATASET_DIR = next((p for p in DATASET_CANDIDATES if p.exists()), PROJECT_ROOT / "dataset")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Output directory
OUTPUT_DIR = Path.cwd() / "output" if (Path.cwd() / "dataset").exists() else PROJECT_ROOT / "output"
MODELS_DIR = BASE_DIR / "models"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Pretrained embedding model
# Open-source, MIT license, < 30M params (strict < 8B param compliance)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Blocking Hyperparameters
TOP_K_TFIDF = 25
TOP_K_EMB = 25
MAX_CANDIDATES_PER_ENTITY = 50

# Training Hyperparameters
RANDOM_STATE = 42
VAL_SIZE = 0.2
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 1000

# F0.5 Optimization Sweep Range
THRESHOLD_START = 0.40
THRESHOLD_END = 0.95
THRESHOLD_STEP = 0.01
DEFAULT_THRESHOLD = 0.75
