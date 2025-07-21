import os
import random
import pickle
import gc
import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from src import config

def compute_lightweight_features_igraph(G):
    """
    Computes lightweight node-level and graph-level features for an igraph graph.
    This version includes checks to prevent NaN or Inf values in graph features.
    """
    node_features_list = []
    if G.vcount() > 0:
        node_degrees = G.degree()
        node_pagerank = G.pagerank()
        node_coreness = G.coreness()
        node_eigenvector_centrality = G.eigenvector_centrality()
        node_closeness_centrality = G.closeness()

        for i in range(G.vcount()):
            node_features = {
                'Degree': node_degrees[i],
                'Pagerank': node_pagerank[i],
                'K-Core': node_coreness[i],
                'Eigenvector': node_eigenvector_centrality[i],
                'Closeness': node_closeness_centrality[i] if node_closeness_centrality[i] is not None else 0.0,
            }
            node_features_list.append(node_features)

    degrees = G.degree()
    assortativity = G.assortativity_degree(directed=False) if G.vcount() > 1 else 0.0
    clustering = G.transitivity_undirected() if G.vcount() > 1 else 0.0
    avg_path_len = G.average_path_length() if G.vcount() > 1 else 0.0

    graph_features_dict = {
        "Degree Variance": np.var(degrees) if degrees else 0.0,
        "Assortativity": assortativity if np.isfinite(assortativity) else 0.0,
        "Density": G.density(),
        "Clustering": clustering if np.isfinite(clustering) else 0.0,
        "Average Path Length": avg_path_len if np.isfinite(avg_path_len) else 0.0,
        "edges": G.ecount(),
        "nodes": G.vcount()
    }
    return node_features_list, graph_features_dict

def generate_single_graph(family, num_nodes):
    """
    Generates a single random graph from a specified family.
    """
    try:
        if family == "erdos_renyi":
            p = random.uniform(0.001, 0.0015)
            G = ig.Graph.Erdos_Renyi(n=num_nodes, p=p)
        elif family == "barabasi_albert":
            m = random.randint(3, 5)
            G = ig.Graph.Barabasi(n=num_nodes, m=m)
        elif family == "watts_strogatz":
            k = random.randint(4, 6)
            p = random.uniform(0.3, 0.6)
            G = ig.Graph.Watts_Strogatz(dim=1, size=num_nodes, nei=k, p=p)
        elif family == "stochastic_block_model":
            num_blocks = random.randint(4, 6)
            block_sizes = [num_nodes // num_blocks] * num_blocks
            for i in range(num_nodes % num_blocks):
                block_sizes[i] += 1
            p_within = np.random.uniform(0.001, 0.005)
            p_between = np.random.uniform(0.0005, 0.002, size=(num_blocks, num_blocks))
            p_matrix = (p_between + p_between.T) / 2
            np.fill_diagonal(p_matrix, p_within)
            G = ig.Graph.SBM(num_nodes, p_matrix.tolist(), block_sizes=block_sizes)
        elif family == "holme_kim":
            m = random.randint(3, 5)
            p = random.uniform(0.2, 0.3)
            G_nx = nx.powerlaw_cluster_graph(num_nodes, m, p)
            if not nx.is_connected(G_nx):
                largest_cc = max(nx.connected_components(G_nx), key=len)
                G_nx = G_nx.subgraph(largest_cc).copy()
            G = ig.Graph.from_networkx(G_nx)
        else:
            return None

        if not G.is_connected():
            G = G.components().giant()
        return G
    except Exception as e:
        print(f"Warning: Graph generation failed for {family} with {num_nodes} nodes. Error: {e}")
        return None


def run_batched_generation():
    """
    Generates and saves graphs and their features in batches.
    """
    print("Starting batched graph generation...")
    global_id = 0
    for family in config.FAMILIES:
        family_idx = config.FAMILIES.index(family)
        graphs_generated = 0
        batch_id = 0
        while graphs_generated < config.TOTAL_GRAPHS_PER_FAMILY:
            batch_graphs, batch_node_features, batch_graph_features = [], [], []

            for _ in range(config.BATCH_SIZE_GEN):
                if graphs_generated >= config.TOTAL_GRAPHS_PER_FAMILY:
                    break

                num_nodes = random.randint(config.MIN_NODES, config.MAX_NODES)
                G = generate_single_graph(family, num_nodes)
                if G is None:
                    continue

                node_features, graph_features = compute_lightweight_features_igraph(G)
                graph_features["family"] = family_idx
                graph_features["graph_id"] = global_id

                batch_graphs.append(G)
                batch_node_features.append(node_features)
                batch_graph_features.append(graph_features)

                graphs_generated += 1
                global_id += 1

            if not batch_graphs:
                batch_id += 1
                continue

            feature_df = pd.DataFrame(batch_graph_features)
            feature_df.to_csv(os.path.join(config.FEATURE_DIR, f"{family}_graph_features_batch_{batch_id:02d}.csv"), index=False)
            with open(os.path.join(config.FEATURE_DIR, f"{family}_node_features_batch_{batch_id:02d}.pkl"), "wb") as f:
                pickle.dump(batch_node_features, f)
            with open(os.path.join(config.GRAPH_DIR, f"{family}_graphs_batch_{batch_id:02d}.pkl"), "wb") as f:
                pickle.dump(batch_graphs, f)

            print(f"Saved batch {batch_id:02d} for {family} ({len(batch_graphs)} graphs)")
            batch_id += 1
            gc.collect()
    print("Graph generation complete.")
