import torch
from torch import Tensor
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv, GATConv
from torch_geometric.nn import global_max_pool, global_mean_pool, SAGPooling

import torch.nn.functional as F

class GCNGraph(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, pooling_layer):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.gcn_conv_layers = nn.ModuleList([GCNConv(in_channels, self.hidden_channels)])
        for i in range(1, num_layers):
            self.gcn_conv_layers.append(GCNConv(self.hidden_channels, self.hidden_channels))

        self.nonlinear_fn = nn.ReLU()
        self.pooling_layer = pooling_layer
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch) -> Tensor:
        #print(x.shape)
        for i, gcn_layer in enumerate(self.gcn_conv_layers):
            x = gcn_layer(x, edge_index)
            x = self.nonlinear_fn(x)

        if isinstance(self.pooling_layer, SAGPooling):
            sag_output = self.pooling_layer(x, edge_index, batch=batch)
            x = sag_output[0]
            edge_index = sag_output[1]
            batch= sag_output[3]
            x = global_mean_pool(x, batch = batch)
        else:
            x = self.pooling_layer(x, batch=batch)
        #print(x.shape)
        #x = F.dropout(x, p=0.2, training=self.training)
        y = self.linear(x)

        return y

    def to_str(self):
        return "{}_embdim={}_{}".format(
            "GCN-CONV",
            self.hidden_channels,
            type(self.pooling_layer).__name__)


class GATGraph(torch.nn.Module):
    def __init__(self, num_layers, in_channels, hidden_channels, num_heads, out_channels, pooling_layer):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.layers = nn.ModuleList([GATv2Conv(in_channels, self.hidden_channels, heads=num_heads, add_self_loops=False)])
        for i in range(1, num_layers):
            self.layers.append(GATv2Conv(self.hidden_channels, self.hidden_channels, heads=num_heads, add_self_loops=False))

        self.nonlinear_fn = nn.ReLU()
        self.pooling_layer = pooling_layer
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, return_attention_weights=False) -> Tensor:
        #print(x.shape)
        attn_weights_per_layer = None
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, edge_index, return_attention_weights=True)
            x = self.nonlinear_fn(x)
            if return_attention_weights:
                if attn_weights_per_layer is None:
                    attn_weights_per_layer = {}
                attn_weights_per_layer[i]=attn_weights

        if isinstance(self.pooling_layer, SAGPooling):
            sag_output = self.pooling_layer(x, edge_index, batch=batch)
            x = sag_output[0]
            edge_index = sag_output[1]
            batch= sag_output[3]
            x = global_mean_pool(x, batch=batch)
        else:
            x = self.pooling_layer(x, batch=batch)
        #print(x.shape)
        #x = F.dropout(x, p=0.2, training=self.training)
        y = self.linear(x)

        if return_attention_weights:
            return y, attn_weights_per_layer

        return y

    def to_str(self):
        if isinstance(self.pooling_layer, SAGPooling):
            pooling_fn = "SAG"
        else:
            pooling_fn = self.pooling_layer.__name__

        return "{}_layers={}_heads={}_embdim={}_{}".format(
            "GAT-CONV",
            len(self.layers),
            self.num_heads,
            self.hidden_channels,
            pooling_fn)
