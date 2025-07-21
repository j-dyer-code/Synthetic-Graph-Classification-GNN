import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv, GINConv, TransformerConv, global_mean_pool
from src import config

class GraphFamilyClassifierBase(nn.Module):
    """
    A base class for Graph Neural Network classifiers.
    """
    def __init__(self, in_channels, hidden_channels, num_classes, graph_feature_dim, dropout_rate):
        super().__init__()
        self.conv1 = self.define_conv_layer_1(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = self.define_conv_layer_2(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc1 = nn.Linear(hidden_channels + graph_feature_dim, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def define_conv_layer_1(self, in_channels, hidden_channels):
        raise NotImplementedError

    def define_conv_layer_2(self, hidden_channels, out_channels):
        raise NotImplementedError

    def forward(self, data):
        """
        Defines the forward pass for the classifier.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        graph_features = data.graph_features.view(x.shape[0], -1)
        x = torch.cat([x, graph_features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.bn3(x)
        embeddings = self.dropout(x)
        logits = self.fc2(embeddings)
        return logits, embeddings

class GraphFamilyClassifierGCN(GraphFamilyClassifierBase):
    def define_conv_layer_1(self, in_channels, hidden_channels): return GCNConv(in_channels, hidden_channels)
    def define_conv_layer_2(self, hidden_channels, out_channels): return GCNConv(hidden_channels, out_channels)

class GraphFamilyClassifierGAT(GraphFamilyClassifierBase):
    def define_conv_layer_1(self, in_channels, hidden_channels): return GATConv(in_channels, hidden_channels, heads=1, concat=True)
    def define_conv_layer_2(self, hidden_channels, out_channels): return GATConv(hidden_channels, out_channels, heads=1, concat=False)

class GraphFamilyClassifierGATV2(GraphFamilyClassifierBase):
    def define_conv_layer_1(self, in_channels, hidden_channels): return GATv2Conv(in_channels, hidden_channels, heads=1, concat=True)
    def define_conv_layer_2(self, hidden_channels, out_channels): return GATv2Conv(hidden_channels, out_channels, heads=1, concat=False)

class GraphFamilyClassifierSAGE(GraphFamilyClassifierBase):
    def define_conv_layer_1(self, in_channels, hidden_channels): return SAGEConv(in_channels, hidden_channels)
    def define_conv_layer_2(self, hidden_channels, out_channels): return SAGEConv(hidden_channels, out_channels)

class GraphFamilyClassifierGIN(GraphFamilyClassifierBase):
    def _make_mlp(self, in_dim, out_dim): return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
    def define_conv_layer_1(self, in_channels, hidden_channels): return GINConv(self._make_mlp(in_channels, hidden_channels), train_eps=True)
    def define_conv_layer_2(self, hidden_channels, out_channels): return GINConv(self._make_mlp(hidden_channels, out_channels), train_eps=True)

class GraphFamilyClassifierGTN(GraphFamilyClassifierBase):
    def define_conv_layer_1(self, in_channels, hidden_channels): return TransformerConv(in_channels, hidden_channels, heads=1, concat=True)
    def define_conv_layer_2(self, hidden_channels, out_channels): return TransformerConv(hidden_channels, out_channels, heads=1, concat=False)

def get_model_from_arch(model_name, num_node_features, num_graph_features, hidden_channels, dropout_rate):
    """Factory function to instantiate a model by name."""
    model_map = {'GCN': GraphFamilyClassifierGCN, 'GAT': GraphFamilyClassifierGAT, 'GATV2': GraphFamilyClassifierGATV2, 'SAGE': GraphFamilyClassifierSAGE, 'GIN': GraphFamilyClassifierGIN, 'GTN': GraphFamilyClassifierGTN}
    model_class = model_map.get(model_name)
    if model_class:
        return model_class(num_node_features, hidden_channels, len(config.FAMILIES), num_graph_features, dropout_rate)
    raise ValueError(f"Unknown model type: {model_name}")
