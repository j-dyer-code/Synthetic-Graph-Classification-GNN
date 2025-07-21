import os
import torch

# --- General Settings ---
DEBUG_MODE = False
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Generation ---
BATCH_SIZE_GEN = 10 if DEBUG_MODE else 20
TOTAL_GRAPHS_PER_FAMILY = 100 if DEBUG_MODE else 400
FAMILIES = ["erdos_renyi", "barabasi_albert", "watts_strogatz", "stochastic_block_model", "holme_kim"]
FAMILY_NAMES_PRETTY = ["ER", "BA", "WS", "SBM", "HK"]
MIN_NODES = 500 if DEBUG_MODE else 5_000
MAX_NODES = 1000 if DEBUG_MODE else 10_000

# --- Model Architectures ---
MODELS = ['GAT', 'GCN', 'SAGE', 'GIN', 'GATV2', 'GTN']

# --- Training and Tuning ---
TRAIN_RATIO = 0.8
BATCH_SIZE_TRAIN = 32
NUM_TRIALS_OPTUNA = 5 if DEBUG_MODE else 50
MAX_EPOCHS_TRAIN = 10 if DEBUG_MODE else 100
EARLY_STOPPING_PATIENCE = 10

# --- Directories ---
# Parent directories for outputs and data
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Subdirectories for specific artifacts
PLOT_DIR = os.path.join(RESULTS_DIR, "evaluation_plots")
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
FINAL_MODELS_DIR = os.path.join(RESULTS_DIR, "final_models")
FEATURE_DIR = os.path.join(DATA_DIR, "graph_features")
GRAPH_DIR = os.path.join(DATA_DIR, "graph_objects")

# Paths for specific files
DATA_PKL_PATH = os.path.join(DATA_DIR, 'pyg_data.pkl')
METRICS_PKL_PATH = os.path.join(METRICS_DIR, 'metrics_dict.pkl')

# --- Visualization ---
CLASS_COLORS = {
    "ER": "#66c2a5",
    "BA": "#fc8d62",
    "WS": "#8da0cb",
    "SBM": "#e78ac3",
    "HK": "#a6d854"
}
