import math

import numpy as np
import torch
import torch_geometric.nn
import torchvision.models
from torch import Tensor
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GAT, DeepGCNLayer, GENConv, BatchNorm
from torch_geometric.nn import global_max_pool, global_mean_pool, SAGPooling
from torch_cluster import knn_graph
import bagnets.pytorchnet
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE

import torch.nn.functional as F
from torch.nn import Sequential as Seq
from .gsat_impl.gsat import init_gsat_model
from .stem_approaches import *
from .vig_pytorch.vig import vig_ti_224_gelu, FFN
from .vig_pytorch.pyramid_vig import pvig_ti_224_gelu
from .wignn.wignn_256 import wignn_ti_256_gelu, wignn_encoder
from .gsat_impl.utils.utils import reshape_into_graph_input
from .iwivig.iwivig import iWiViG
from .bagnet_impl import bagnet17

class NonGraphModelWrapper(nn.Module):
    def __init__(self, model, model_str):
        super().__init__()
        self.model = model
        self.model_str = model_str
        self.use_patch_predictions = False

    def forward(self, x):
        output = self.model(x)
        return {"prediction": output}
    def print(self):
        return self.model_str
    

class BagNet17Encoder(nn.Module):
    def __init__(self, encoder_emb_dim, output_emb_dim):
        super().__init__()
        self.model = bagnet17(num_classes=encoder_emb_dim, avg_pool=False)
        self.downsample = nn.Sequential(
            nn.Conv2d(encoder_emb_dim, output_emb_dim, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(output_emb_dim, momentum=0.1))

    def forward(self, x):
        output = self.model(x).permute(0, 3, 1, 2)
        output = self.downsample(output)
        return {"image_feature_map": output}
        
    def print(self):
        return "BagNet17Encoder"

class GraphTransform(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768):
        super().__init__()
        conv = GENConv(in_channels, hidden_channels, aggr='softmax',
                       t=1.0, learn_t=True, num_layers=2)
        self.gnn_layer = DeepGCNLayer(conv, dropout=0.1)
        #self.ffn = FFN(hidden_channels, hidden_channels * 4, act="relu", drop_path=0.1)

    def forward(self, x, edge_index, batch) -> Tensor:
        node_features = self.gnn_layer(x, edge_index)
        return node_features

class DeepGCNGNN(torch.nn.Module):
    def __init__(self, in_channels=768, num_layers=3, hidden_channels=768, num_classes=1):
        super().__init__()
        self.gnn_layers = nn.ModuleList([GraphTransform(in_channels=in_channels, hidden_channels=hidden_channels)
            for _ in range (num_layers)])
        self.prediction = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch) -> Tensor:
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(x, edge_index, batch)

        node_features = global_mean_pool(node_features, batch)
        y = self.prediction(node_features)
        return y

    def print(self):
        return "DeepGCN"

class GSATViG(torch.nn.Module):

    def __init__(self, stem_model, total_num_nodes, num_layers=6, hidden_channels=192, num_classes=1, gsat_r=0.7):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.stem_model = stem_model
        self.total_num_nodes = total_num_nodes
        self.gsat_r = gsat_r

        self.pos_embed = nn.Parameter(torch.zeros(1,
                                                  hidden_channels,
                                                  int(math.sqrt(total_num_nodes)),
                                                  int(math.sqrt(total_num_nodes))))
        
        self.gnn_model = init_gsat_model(hidden_channels=hidden_channels, num_classes=num_classes, dropout_p=0.2)
 
        self.fc2 = FFN(hidden_channels, hidden_channels * 4, act="gelu", drop_path=0)
        self.pooling_layer = nn.AdaptiveAvgPool2d(output_size=1)

        self.prediction = Seq(nn.Conv2d(hidden_channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer("gelu"),
                              nn.Dropout(0),
                              nn.Conv2d(1024, num_classes, 1, bias=True))


    def forward(self, x) -> [Tensor, Tensor]:
        x = self.stem_model(x) + self.pos_embed
        B, C, H, W = x.shape

        graph_data = self.extract_graph(x)
        edge_att, node_embeddings = self.gnn_model(graph_data)

        node_embeddings = node_embeddings.reshape(B, -1, H, W).contiguous()
        node_embeddings = self.fc2(node_embeddings)

        graph_embedding = self.pooling_layer(node_embeddings)
        prediction = self.prediction(graph_embedding).squeeze(-1).squeeze(-1)
        return edge_att, prediction

    def extract_graph(self, x):
        with torch.no_grad():
            node_features, batch = reshape_into_graph_input(x)
            edge_index = knn_graph(node_features, k=5, batch=batch)
            graph_data = Data(node_features, edge_index, batch=batch)
            return graph_data
    def print(self):
        return "{}_{}_layers={},r={}".format(self.stem_model.print(), self.gnn_model.print(), self.num_layers, self.gsat_r)


class GSATSubgraph(torch.nn.Module):
    def __init__(self, num_layers=1, hidden_channels=384, num_classes=1, final_r=0.7, lambda_gsat_loss=1.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers    
        self.gnn_model = init_gsat_model(hidden_channels=hidden_channels, num_layers=num_layers, num_classes=num_classes, gsat_r=final_r, lambda_gsat_loss=lambda_gsat_loss, learn_edge_att=True)

        self.fc1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels),
        )
        self.fc2 = FFN(hidden_channels, hidden_channels * 4, act="gelu", drop_path=0)

        # initialized from the ViG model
        self.k = None

    def forward(self, x) -> [Tensor, Tensor]:
        B, C, H, W = x.shape
        x = self.fc1(x)
        node_features, batch = reshape_into_graph_input(x)
        node_features = F.normalize(node_features, p=2.0, dim=1)
        with torch.no_grad():
            edge_index = knn_graph(node_features, k=self.k, batch=batch)
        graph_data = Data(node_features, edge_index, batch=batch)
        edge_att, node_embeddings = self.gnn_model(graph_data)
        node_embeddings = node_embeddings.reshape(B, -1, H, W).contiguous()
        node_embeddings = self.fc2(node_embeddings)
        return edge_att, node_embeddings

def init_model(stem_approach,
               stem_channels,
               gnn_model,
               image_size,
               num_classes,
               encoder_model="WIGNN",
               encoder_downsample_wo_overlap=True,
               encoder_backbone_wig_blocks=3,
               encoder_window_size=8,
               encoder_graph_conv='mr',
               graph_bottleneck_layers=2, 
               gsat_r=0.7,
               lambda_gsat_loss=1.0,
               use_patch_predictions=False,
               learn_edge_att=True):
    print("Encoder wig backbone blocks: {}".format(encoder_backbone_wig_blocks))
    stem_model, num_nodes = resolve_stem_module(stem_approach, stem_channels)
    if gnn_model == "vig":
        model = vig_ti_224_gelu(stem_layer=stem_model, num_nodes=num_nodes, num_classes=num_classes, image_size=image_size)
    elif gnn_model == "pvig":
        model = pvig_ti_224_gelu(stem_layer=stem_model, num_nodes=num_nodes, num_classes=num_classes, image_size=image_size, downsample_wo_overlap=encoder_downsample_wo_overlap)
    elif gnn_model == "WIGNN":
        model = wignn_ti_256_gelu(num_classes=num_classes, use_shifts=False)
    elif gnn_model == "iWiViG":
        k=9
        emb_dim = 384
        if encoder_model == "WIGNN":
            visual_encoder = wignn_encoder(window_size=encoder_window_size, backbone_wig_blocks=encoder_backbone_wig_blocks, downsample_wo_overlap=encoder_downsample_wo_overlap, graph_conv=encoder_graph_conv, img_size=image_size)
        elif encoder_model == "vig":
            emb_dim=192
            k=18
            visual_encoder = vig_ti_224_gelu(stem_layer=stem_model, num_nodes=num_nodes, num_classes=num_classes, image_size=image_size, encoder_only=True, n_blocks=10)
        elif encoder_model == "pvig":
            visual_encoder = pvig_ti_224_gelu(stem_layer=stem_model, num_nodes=num_nodes, num_classes=num_classes, image_size=image_size, downsample_wo_overlap=False, encoder_only=True, backbone_wig_blocks=3)
        else:
            visual_encoder = BagNet17Encoder(192, emb_dim)
        model = iWiViG(visual_encoder, emb_dim, num_layers=graph_bottleneck_layers, num_classes=num_classes, gsat_r=gsat_r, lambda_gsat_loss=lambda_gsat_loss, k=k, use_patch_predictions=use_patch_predictions, learn_edge_att=learn_edge_att)
    elif gnn_model == "resnet":
        model = NonGraphModelWrapper(torchvision.models.resnet50(num_classes=num_classes), "resnet50")
    elif gnn_model == "vit":
        model = NonGraphModelWrapper(torchvision.models.vit_b_16(num_classes=num_classes), "vit_b_16")
    else:
        model = NonGraphModelWrapper(bagnets.pytorchnet.bagnet33(num_classes=num_classes), "bagnet33")

    model_label = model.print()

    return model, model_label