import os
import random
import numpy as np
import torch
from . import config

def setup_environment():
    """Set up directories and random seeds for reproducibility."""
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    os.makedirs(config.GRAPH_DIR, exist_ok=True)
    os.makedirs(config.FINAL_MODELS_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"Using device: {config.DEVICE}")
