# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear, Sequential as Seq
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import DeepGCNLayer

from .conv_layers import GINConv, GINEConv

class GINGrapherLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.node_encoder = Linear(self.hidden_size, self.hidden_size)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)
        self.graph_conv = GINConv(Seq(
            GIN.MLP(self.hidden_size, self.hidden_size)
        ))
        self.fc2 = FFN(self.hidden_size, self.hidden_size*2)
        self.ffn = FFN(self.hidden_size, self.hidden_size * 4)

    def forward(self, x, edge_index, edge_attr=None, edge_atten=None):
        _tmp = x
        x = self.batch_norm(self.node_encoder(x))
        x = self.graph_conv(x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
        x = self.fc2(x)
        x = _tmp + x
        return self.ffn(x)

class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)
        self.use_residuals = model_config["use_residuals"]
        self.residuals_mode = model_config["residuals_mode"]
        self.use_pooling = model_config["pooling"]

        if edge_attr_dim != 0 and self.use_edge_attr:
            self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINGrapherLayer(hidden_size))


    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):

        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
        return x

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.get_emb(x, edge_index, batch, edge_attr, edge_atten)
        return x

    def print(self):
        output_str = "GIN_CONV-{}_layers".format(self.n_layers)
        if self.use_residuals:
            output_str = output_str + "_with_{}_residuals".format(self.residuals_mode)
        return output_str


    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            Linear(out_channels, out_channels),
        )

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + shortcut

if __name__ == '__main__':
    model_config = {'model_name': 'GIN', 'hidden_size': 192, 'n_layers': 3,
                    'dropout_p': 0.2, 'use_edge_attr': False, "use_residuals": False,
                    "residuals_mode": None, "pooling": False}
    model = GIN(192, None, 10, False, model_config)
    print(model)