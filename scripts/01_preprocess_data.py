# scripts/01_preprocess_data.py
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.feature_engineering import load_and_preprocess_dataset
from src.utils import setup_environment

if __name__ == '__main__':
    print("--- Step 1: Setting up environment and preprocessing data ---")
    # Setup environment to ensure directories exist, but data generation is separate
    setup_environment()
    
    # This function now handles the entire preprocessing pipeline
    load_and_preprocess_dataset()
    
    print("--- Data preprocessing and feature pruning complete ---")
