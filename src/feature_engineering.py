import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from src import config

def get_important_features(features_list_of_dicts, labels):
    """
    Identifies important features using a Random Forest Classifier ensemble.
    """
    if not features_list_of_dicts:
        return []
    feature_names = list(features_list_of_dicts[0].keys())
    if not feature_names:
        return []

    all_feature_values = [[graph_features[name] for name in feature_names] for graph_features in features_list_of_dicts]
    
    n_runs = 5
    importances = np.zeros((n_runs, len(feature_names)))
    for i in range(n_runs):
        rf = RandomForestClassifier(random_state=i)
        rf.fit(all_feature_values, labels)
        importances[i, :] = rf.feature_importances_

    mean_importances = np.mean(importances, axis=0)
    
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_importances})
    sorted_feature_df = feature_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    sorted_feature_df['Cumulative Importance'] = sorted_feature_df['Importance'].cumsum()
    
    print("Feature Importance Report:")
    display(sorted_feature_df)

    threshold = 0.8
    important_df = sorted_feature_df[sorted_feature_df['Cumulative Importance'] >= threshold]
    
    if not important_df.empty:
        cutoff_index = important_df.index[0]
        important_feature_names = sorted_feature_df.iloc[:cutoff_index + 1]['Feature'].tolist()
    else:
        important_feature_names = sorted_feature_df['Feature'].tolist()

    return important_feature_names

def get_pruned_features(features_list_of_dicts, important_feature_names):
    """
    Filters a list of feature dictionaries to keep only important features.
    """
    if not features_list_of_dicts:
        return []
    
    unimportant_features = set(features_list_of_dicts[0].keys()) - set(important_feature_names)
    print(f"Pruning features: {list(unimportant_features)}")

    pruned_features = []
    for graph_features in features_list_of_dicts:
        pruned_graph = {name: graph_features[name] for name in important_feature_names if name in graph_features}
        pruned_features.append(pruned_graph)
    
    return pruned_features

def visualise_features(features, labels, level):
    """
    Generates and saves plots to visualize feature distributions and correlations.
    """
    if not features:
        print("No features to visualize.")
        return

    feature_names = list(features[0].keys())
    df_list = []
    for i, feature_dict in enumerate(features):
        family = config.FAMILY_NAMES_PRETTY[labels[i]]
        for feature_name, value in feature_dict.items():
            df_list.append({'Value': value, 'Feature': feature_name, 'Family': family})
    df = pd.DataFrame(df_list)

    unique_features = df['Feature'].unique()
    g = sns.FacetGrid(data=df, col='Feature', hue='Family', palette="Set2", col_wrap=len(unique_features), height=4, aspect=1.2, sharey=False, sharex=False, legend_out=True)
    g.map(sns.kdeplot, 'Value', fill=True, alpha=0.5)
    handles, labels = g.axes[0].get_legend_handles_labels()
    g.figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(config.FAMILIES), title='Family', frameon=False)
    plt.savefig(os.path.join(config.PLOT_DIR, f"{level}_kde.pdf"), bbox_inches='tight')
    plt.show()

    g_box = sns.FacetGrid(data=df, col='Feature', hue='Family', palette="Set2", col_wrap=len(unique_features), height=4, aspect=1.2, sharey=False, sharex=False)
    g_box.map(sns.boxplot, 'Family', 'Value')
    plt.savefig(os.path.join(config.PLOT_DIR, f"{level}_box.pdf"), bbox_inches='tight')
    plt.show()

    df_pivot = df.pivot_table(index=df.index // len(feature_names), columns='Feature', values='Value')
    corr = df_pivot.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, annot_kws={"size": 26})
    plt.title(f"{level.capitalize()} Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{level}_feature_correlation.pdf"), bbox_inches='tight')
    plt.show()

def prune_features(features_list_of_dicts, labels, level):
    """
    Iteratively prunes features until the set of important features stabilizes.
    """
    if not features_list_of_dicts:
        return [], 0
    
    current_features = features_list_of_dicts
    
    while True:
        num_before_pruning = len(current_features[0])
        important_names = get_important_features(current_features, labels)
        current_features = get_pruned_features(current_features, important_names)
        num_after_pruning = len(current_features[0])
        
        print(f"Pruning Pass ({level}): {num_before_pruning} -> {num_after_pruning} features.")
        if num_after_pruning == num_before_pruning:
            break

    if current_features:
        visualise_features(current_features, labels, level)
        return current_features, len(current_features[0])
    
    print(f"All {level} features were pruned.")
    return [], 0

def convert_to_pyg(graphs, all_node_features, important_node_feature_names, pruned_graph_features, labels):
    """
    Converts igraph graphs and features into PyTorch Geometric Data objects.
    """
    pyg_data = []
    for i, G_ig in enumerate(graphs):
        try:
            edge_list = G_ig.get_edgelist()
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            node_feature_tensors = []
            if all_node_features[i] and important_node_feature_names:
                for node_feat_dict in all_node_features[i]:
                    node_feat = [node_feat_dict.get(name, 0.0) for name in important_node_feature_names]
                    node_feature_tensors.append(node_feat)
            
            data_x = torch.tensor(node_feature_tensors, dtype=torch.float) if node_feature_tensors else torch.empty((G_ig.vcount(), len(important_node_feature_names)), dtype=torch.float)

            graph_feat_values = list(pruned_graph_features[i].values()) if pruned_graph_features and pruned_graph_features[i] else []
            data_graph_features = torch.tensor(graph_feat_values, dtype=torch.float)

            data = Data(x=data_x, edge_index=edge_index, y=torch.tensor(labels[i], dtype=torch.long), graph_features=data_graph_features, num_nodes=G_ig.vcount())
            pyg_data.append(data)
        except Exception as e:
            print(f"Error converting graph {i} to PyG: {e}")
            continue
            
    random.shuffle(pyg_data)
    return pyg_data

def load_and_preprocess_dataset():
    """
    Loads generated data, prunes features, and converts to PyG format.
    """
    print("Loading and preprocessing dataset...")
    all_graphs, all_node_features, all_graph_features, all_labels = [], [], [], []
    for family in config.FAMILIES:
        batch_id = 0
        while True:
            graph_feature_file = os.path.join(config.FEATURE_DIR, f"{family}_graph_features_batch_{batch_id:02d}.csv")
            node_feature_file = os.path.join(config.FEATURE_DIR, f"{family}_node_features_batch_{batch_id:02d}.pkl")
            graph_file = os.path.join(config.GRAPH_DIR, f"{family}_graphs_batch_{batch_id:02d}.pkl")

            if not all(os.path.exists(f) for f in [graph_feature_file, node_feature_file, graph_file]):
                break

            graph_feature_df = pd.read_csv(graph_feature_file)
            all_graph_features.extend(graph_feature_df.drop(['family', 'graph_id', 'nodes','edges'], axis=1).to_dict('records'))
            all_labels.extend(graph_feature_df['family'].tolist())
            with open(node_feature_file, "rb") as f:
                all_node_features.extend(pickle.load(f))
            with open(graph_file, "rb") as f:
                all_graphs.extend(pickle.load(f))
            batch_id += 1
    
    print(f"Loaded a total of {len(all_graphs)} graphs.")

    node_features_for_pruning = []
    if all_node_features and all_node_features[0]:
        feature_keys = all_node_features[0][0].keys()
        for graph_node_features in all_node_features:
            mean_features = {key: np.mean([nf[key] for nf in graph_node_features]) for key in feature_keys} if graph_node_features else {key: 0.0 for key in feature_keys}
            node_features_for_pruning.append(mean_features)

    print("\n--- Starting Node-Level Feature Pruning ---")
    pruned_node_features_means, num_node_features = prune_features(node_features_for_pruning, all_labels, level='node')
    
    print("\n--- Starting Graph-Level Feature Pruning ---")
    pruned_graph_features, num_graph_features = prune_features(all_graph_features, all_labels, level='graph')

    important_node_feature_names = list(pruned_node_features_means[0].keys()) if pruned_node_features_means else []
    print(f"\nRetained {num_node_features} node features: {important_node_feature_names}")
    print(f"Retained {num_graph_features} graph features: {list(pruned_graph_features[0].keys()) if pruned_graph_features else []}")

    pyg_data = convert_to_pyg(all_graphs, all_node_features, important_node_feature_names, pruned_graph_features, all_labels)
    
    with open(config.DATA_PKL_PATH, 'wb') as f:
        pickle.dump(pyg_data, f)
    print(f"\nPreprocessing complete. Saved {len(pyg_data)} PyG data objects to {config.DATA_PKL_PATH}.")

    return pyg_data, num_node_features, num_graph_features

def prepare_dataloaders(pyg_data):
    """
    Splits PyG data into training and testing sets and creates DataLoaders.
    """
    num_train = int(len(pyg_data) * config.TRAIN_RATIO)
    train_data, test_data = pyg_data[:num_train], pyg_data[num_train:]

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE_TRAIN, shuffle=False)
    
    print(f"Data split into {len(train_data)} training and {len(test_data)} testing samples.")
    return train_loader, test_loader
