import sys, os, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, accuracy_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display

from src import config
from src.models import get_model_from_arch
from src.feature_engineering import prepare_dataloaders

def visualise_embeddings(model, name, dataloader, device):
    """
    Generates and saves 2D t-SNE and 3D UMAP visualizations of graph embeddings.
    """
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            _, emb = model(data)
            embeddings.append(emb.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    embeddings, labels = np.vstack(embeddings), np.hstack(labels)
    label_names = [config.FAMILY_NAMES_PRETTY[l] for l in labels]

    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=config.SEED)
    tsne_result = tsne.fit_transform(embeddings)
    df_tsne = pd.DataFrame({"x": tsne_result[:, 0], "y": tsne_result[:, 1], "label": label_names})
    plt.figure(figsize=(12, 9))
    sns.scatterplot(data=df_tsne, x="x", y="y", hue="label", hue_order=config.FAMILY_NAMES_PRETTY, palette=config.CLASS_COLORS, alpha=0.9, s=70)
    plt.title(f"t-SNE Embedding Visualization ({name})", fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{name}_tsne.pdf"), bbox_inches='tight')
    plt.show()

    reducer = umap.UMAP(n_neighbors=20, min_dist=0.5, n_components=3, random_state=config.SEED, spread=2.5)
    umap_result = reducer.fit_transform(embeddings)
    label_colors = [config.CLASS_COLORS[label] for label in label_names]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], c=label_colors, alpha=0.9, s=50)
    ax.set_title(f"3D UMAP Embedding Visualization ({name})", fontsize=26)
    ax.set_box_aspect([np.ptp(umap_result[:, i]) for i in range(3)])
    handles = [plt.Line2D([], [], marker='o', color='w', label=label, markerfacecolor=config.CLASS_COLORS[label], markersize=15) for label in config.FAMILY_NAMES_PRETTY]
    ax.legend(handles=handles, title="Graph Family", loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{name}_umap.pdf"), bbox_inches='tight')
    plt.show()

def comprehensive_evaluation(model, name, loader, device, metrics_entry):
    """ Performs a full evaluation of a model and generates all relevant plots and reports. """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(probs.argmax(axis=1))
            y_prob.extend(probs)
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    print(f"\n--- Comprehensive Evaluation for {name} ---")
    cm = confusion_matrix(y_true, y_pred, normalize='pred')
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=config.FAMILY_NAMES_PRETTY, yticklabels=config.FAMILY_NAMES_PRETTY, annot_kws={"size":26})
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, f"{name}_confusion_matrix.pdf"))
    plt.show()

    report_df = pd.DataFrame(classification_report(y_true, y_pred, target_names=config.FAMILY_NAMES_PRETTY, digits=4, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(config.METRICS_DIR,f"{name}_classification_report.csv"))
    display(report_df)
    
    visualise_embeddings(model, name, loader, device)

def run_svm_baseline(pyg_data):
    """ Trains and evaluates a baseline SVM classifier on the graph features."""
    print("\n--- Running SVM Baseline ---")
    X, y = [], []
    for data in pyg_data:
        node_feats = torch.mean(data.x, dim=0).cpu().numpy() if data.x.numel() > 0 else np.zeros(data.x.shape[-1] if data.x.numel() > 0 else 0)
        graph_feats = data.graph_features.cpu().numpy() if data.graph_features.numel() > 0 else np.array([])
        combined = np.concatenate([node_feats, graph_feats])
        X.append(combined)
        y.append(data.y.item())
    X, y = np.vstack(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-config.TRAIN_RATIO), stratify=y, random_state=config.SEED)

    svm = SVC(kernel="rbf", C=1.0, random_state=config.SEED, probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=config.FAMILY_NAMES_PRETTY, digits=4, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(config.METRICS_DIR,"SVM_classification_report.csv"))
    display(report_df)

def run_full_evaluation():
    """ Loads all trained models and metrics to run comprehensive evaluations and comparisons."""
    print("\n" + "="*50 + "\n      RUNNING FULL MODEL EVALUATION AND ANALYSIS\n" + "="*50 + "\n")
    with open(config.DATA_PKL_PATH, 'rb') as f: pyg_data = pickle.load(f)
    with open(config.METRICS_PKL_PATH, 'rb') as f: metrics_dict_data = pickle.load(f)
    
    _, test_loader = prepare_dataloaders(pyg_data)
    best_trials_df = pd.read_csv(os.path.join(config.METRICS_DIR,'best_trials_summary.csv'))
    
    metrics_map = {arch: metrics for arch, metrics in zip(metrics_dict_data['Architecture'], metrics_dict_data['Metrics'])}

    for _, row in best_trials_df.iterrows():
        arch = row['Architecture']
        model_path = os.path.join(config.FINAL_MODELS_DIR, f"{arch}_best.pt")
        if not os.path.exists(model_path): continue
        
        num_node_features = test_loader.dataset[0].x.shape[1]
        num_graph_features = test_loader.dataset[0].graph_features.shape[0]
        model = get_model_from_arch(model_name=arch, num_node_features=num_node_features, num_graph_features=num_graph_features, hidden_channels=row['Hidden Channels'], dropout_rate=row['Dropout Rate']).to(config.DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))

        comprehensive_evaluation(model=model, name=arch, loader=test_loader, device=config.DEVICE, metrics_entry=metrics_map[arch])
    
    run_svm_baseline(pyg_data)

if __name__ == '__main__':
    run_full_evaluation()
    print(f"\nFull evaluation complete. Metrics saved to {config.METRICS_DIR}. Plots saved to {config.PLOT_DIR}")
