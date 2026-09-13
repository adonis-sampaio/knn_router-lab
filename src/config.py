"""Central configuration for the kNN Router Lab."""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SUPPORT_SET_PATH = DATA_DIR / "support_set.jsonl"
MODEL_POOL_PATH = DATA_DIR / "model_pool.json"

# Embedding model (small, CPU-friendly, permissive licence)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Routing
SEED = 42
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.70, 0.10, 0.20
K_NEIGHBORS = 100          # paper's best k for utility prediction
LAMBDA_PRESETS = {         # paper §4.3: normalized by c_max
    "Low-cost": 1.0,
    "Balanced": 0.5,
    "High-performance": 0.1,
}
