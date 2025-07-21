import sys, os, gc, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import config
from src.models import get_model_from_arch
from src.training_utils import final_training_loop
from src.feature_engineering import load_and_preprocess_dataset, prepare_dataloaders

def run_final_training(best_trials_df, num_node_features, num_graph_features, train_loader, test_loader):
    """
    Trains the best version of each model architecture found during tuning.
    """
    print("\n--- Starting Final Model Training ---")
    metrics_dict = {'Architecture': [], 'Metrics': []}

    for _, row in best_trials_df.iterrows():
        arch = row['Architecture']
        params = row.to_dict()
        print(f"\nTraining best {arch} model with params: {params}")

        model = get_model_from_arch(model_name=arch, num_node_features=num_node_features, num_graph_features=num_graph_features, hidden_channels=params['Hidden Channels'], dropout_rate=params['Dropout Rate']).to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['Learning Rate'])
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        results = final_training_loop(model, arch, optimizer, scheduler, criterion, train_loader, test_loader, config.DEVICE)
        metrics_dict['Architecture'].append(arch)
        metrics_dict['Metrics'].append(results)
        
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    with open(config.METRICS_PKL_PATH, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print(f"\nFinal training complete. Metrics saved to {config.METRICS_PKL_PATH}.")

if __name__ == '__main__':
    best_trials_df = pd.read_csv(os.path.join(config.METRICS_DIR,'best_trials_summary.csv'))
    
    # Load the single preprocessed data file
    print("--- Loading preprocessed data from pyg_data.pkl ---")
    with open(config.DATA_PKL_PATH, 'rb') as f:
        pyg_data = pickle.load(f)

    # Get feature dimensions from the loaded data
    if not pyg_data:
        raise ValueError("No data found in pyg_data.pkl. Please run preprocessing first.")
        
    num_node_features = pyg_data[0].num_node_features
    num_graph_features = pyg_data[0].graph_features.shape[0] if pyg_data[0].graph_features is not None and pyg_data[0].graph_features.numel() > 0 else 0

    print(f"Loaded data with {num_node_features} node features and {num_graph_features} graph features.")

    # Prepare dataloaders and run final training
    train_loader, test_loader = prepare_dataloaders(pyg_data)
    run_final_training(best_trials_df, num_node_features, num_graph_features, train_loader, test_loader)
