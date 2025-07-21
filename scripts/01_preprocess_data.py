# scripts/01_preprocess_data.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import load_and_preprocess_dataset
from src.utils import setup_environment

if __name__ == '__main__':
    print("--- Step 1: Setting up environment and preprocessing data ---")
    # Setup environment to ensure directories exist, but data generation is separate
    setup_environment()
    
    # This function now handles the entire preprocessing pipeline
    load_and_preprocess_dataset()
    
    print("--- Data preprocessing and feature pruning complete ---")
