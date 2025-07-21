import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import setup_environment
from src.data_generation import run_batched_generation

if __name__ == '__main__':
    print("--- Step 1: Setting up environment and generating data ---")
    setup_environment()
    run_batched_generation()
    print("--- Data generation complete ---")
