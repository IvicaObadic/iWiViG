import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, sort_edge_index

def reorder_like(from_edge_index, to_edge_index, values):
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
    return values[ranking]

def reshape_into_graph_input(x):
    B, C, H, W = x.shape
    node_features = x.reshape(B, C, -1).transpose(1, 2)
    B, num_nodes, num_features = node_features.shape
    node_features = node_features.reshape(((B * num_nodes), num_features))
    batch = [int(i / num_nodes) for i in range(node_features.shape[0])]
    batch = torch.tensor(batch, dtype=torch.int64).cuda()
    return node_features, batch