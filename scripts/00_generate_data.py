import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import setup_environment
from src.data_generation import run_batched_generation

if __name__ == '__main__':
    print("--- Step 1: Setting up environment and generating data ---")
    setup_environment()
    run_batched_generation()
    print("--- Data generation complete ---")
