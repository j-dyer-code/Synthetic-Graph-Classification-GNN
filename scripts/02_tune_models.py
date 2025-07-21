import sys, os, time, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from IPython.display import display
import pickle

from src import config
from src.models import get_model_from_arch
from src.training_utils import evaluate_one_epoch
from src.feature_engineering import load_and_preprocess_dataset, prepare_dataloaders

def objective(trial, model_name, num_node_features, num_graph_features, train_loader, test_loader):
    """ Optuna objective function for hyperparameter tuning."""
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 96, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    try:
        model = get_model_from_arch(model_name, num_node_features, num_graph_features, hidden_channels, dropout_rate).to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        start = time.time()
        for epoch in range(10): 
            model.train()
            for data in train_loader:
                data = data.to(config.DEVICE)
                optimizer.zero_grad()
                out, _ = model(data)
                loss = criterion(out, data.y)
                if torch.isnan(loss): raise optuna.exceptions.TrialPruned()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            val_loss, val_acc = evaluate_one_epoch(model, test_loader, criterion, config.DEVICE)
            if np.isnan(val_loss): raise optuna.exceptions.TrialPruned()

            scheduler.step(val_loss)
            trial.report(val_loss, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        end = time.time()
        
        trial.set_user_attr("val_accuracy", val_acc)
        trial.set_user_attr("model_name", model_name)
        trial.set_user_attr("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))
        trial.set_user_attr("training_time", end-start)
        return val_loss

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"Trial {trial.number} ran out of memory and was pruned.")
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else: raise e

def run_hyperparameter_tuning(num_node_features, num_graph_features, train_loader, test_loader):
    """ Performs hyperparameter tuning for all GNN models using Optuna."""
    print("\n--- Starting Hyperparameter Tuning ---")
    best_trials = []
    for model_name in config.MODELS:
        print(f"\nTuning {model_name}...")
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), sampler=optuna.samplers.TPESampler(seed=config.SEED))
        study.optimize(lambda trial: objective(trial, model_name, num_node_features, num_graph_features, train_loader, test_loader), n_trials=config.NUM_TRIALS_OPTUNA, show_progress_bar=True)
        best_trials.append(study.best_trial)

    trial_data = [{'Architecture': t.user_attrs['model_name'], 'Validation Loss': t.value, 'Validation Accuracy': t.user_attrs['val_accuracy'], 'Hidden Channels': t.params['hidden_channels'], 'Learning Rate': t.params['learning_rate'], 'Dropout Rate': t.params['dropout_rate'], 'Training Time': t.user_attrs['training_time'], 'Model Size': t.user_attrs['num_params']} for t in best_trials]
    best_trials_df = pd.DataFrame(trial_data).sort_values(by='Validation Loss').reset_index(drop=True)
    best_trials_df.to_csv(os.path.join(config.METRICS_DIR,'best_trials_summary.csv'), index=False)
    print("\nHyperparameter tuning complete. Best trials summary:")
    display(best_trials_df)
    return best_trials_df


if __name__ == '__main__':
    # Load the single preprocessed data file
    print("--- Loading preprocessed data from pyg_data.pkl ---")
    with open(config.DATA_PKL_PATH, 'rb') as f:
        pyg_data = pickle.load(f)

    # Get feature dimensions from the loaded data
    if not pyg_data:
        raise ValueError("No data found in pyg_data.pkl. Please run preprocessing first.")
    
    # Safely get feature dimensions from the first data object
    num_node_features = pyg_data[0].num_node_features
    num_graph_features = pyg_data[0].graph_features.shape[0] if pyg_data[0].graph_features is not None and pyg_data[0].graph_features.numel() > 0 else 0
    
    print(f"Loaded data with {num_node_features} node features and {num_graph_features} graph features.")

    # Prepare dataloaders and run tuning
    train_loader, test_loader = prepare_dataloaders(pyg_data)
    run_hyperparameter_tuning(num_node_features, num_graph_features, train_loader, test_loader)
