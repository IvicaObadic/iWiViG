import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.pool import SAGPooling

from torch_geometric.data import Data

from torch_cluster import knn_graph

from models.gsat_impl.gsat import init_gsat_model
from models.gsat_impl.utils.utils import reshape_into_graph_input
from models.vig_pytorch.vig import act_layer
from models.wignn.wignngcnlib.torch_edge import DenseDilatedKnnGraph


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.BatchNorm1d(out_features),
        )

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x + shortcut
        return x


class GraphBottleneckPredictionWrapper(nn.Module):
    def __init__(self, gnn_model, hidden_channels, pred_emb_dim, num_classes):
        super().__init__()
        self.gnn_model = gnn_model
        self.pooling_layer = global_add_pool
        #self.pooling_layer = SAGPooling(hidden_channels, ratio=0.6)
        self.prediction = Seq(nn.Linear(hidden_channels, pred_emb_dim, bias=True),
                              nn.BatchNorm1d(pred_emb_dim),
                              act_layer("gelu"),
                              nn.Dropout(0),
                              nn.Linear(pred_emb_dim, num_classes, bias=True))

    

    def forward(self, node_embeddings, edge_index, batch, edge_attr=None):
        graph_input_data = Data(x=node_embeddings, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        raw_edge_att, edge_att, node_embeddings_graph_processing = self.gnn_model(graph_input_data)

        if isinstance(self.pooling_layer, SAGPooling):
            pooling_result = self.pooling_layer(node_embeddings_graph_processing, edge_index, batch=batch)
            node_embeddings_graph_processing = pooling_result[0]
            batch = pooling_result[3]
            graph_embedding = global_add_pool(node_embeddings_graph_processing, batch)
        else:
            graph_embedding = self.pooling_layer(node_embeddings_graph_processing, batch)
        prediction = self.prediction(graph_embedding).squeeze(-1)

        return {"edge_att": edge_att,
                "raw_edge_att": raw_edge_att,
                "node_embeddings": node_embeddings, 
                "node_embeddings_graph_processing": node_embeddings_graph_processing, 
                "prediction": prediction}


class iWiViG(nn.Module):
    def __init__(self, visual_encoder, hidden_channels, num_layers, num_classes, gsat_r, lambda_gsat_loss, pred_emb_dim=1024, dilation=2, k=9, use_patch_predictions=True, learn_edge_att=True):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.dilation = dilation
        self.k=k
        self.gsat = init_gsat_model(hidden_channels=hidden_channels, num_layers=num_layers, num_classes=num_classes, gsat_r=gsat_r, lambda_gsat_loss=lambda_gsat_loss, dropout_p=0.3, learn_edge_att=learn_edge_att)
        self.gnn_bottleneck = GraphBottleneckPredictionWrapper(self.gsat, hidden_channels, pred_emb_dim, num_classes)
        self.use_patch_predictions = use_patch_predictions

        self.patch_prediction = Seq(nn.Conv2d(hidden_channels, num_classes, 1, bias=True))
        self.model_init()
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    def image_to_graph(self, image):
        image_feature_map = self.visual_encoder(image)["image_feature_map"]
        print("Feature map shape: {}".format(image_feature_map.shape))
        self.B = image_feature_map.shape[0]
        self.C = image_feature_map.shape[1]
        self.H = image_feature_map.shape[2]
        self.W = image_feature_map.shape[3]
        patch_predictions = None
        if self.use_patch_predictions:
            patch_predictions = self.patch_prediction(image_feature_map).squeeze(-1).squeeze(-1)
        # node_features_knn = image_feature_map.reshape(self.B, self.C, -1).unsqueeze(-1).contiguous()
        # edge_index = self.dense_dilated_knn_graph(node_features_knn)
        # edge_index = edge_index.reshape(2, -1)
        node_features, batch = reshape_into_graph_input(image_feature_map)
        node_features = F.normalize(node_features, p=2.0, dim=1)
        with torch.no_grad():
            edge_index = knn_graph(node_features, k=self.k, batch=batch)
        graph_data = Data(node_features, edge_index, batch=batch)
        return graph_data, patch_predictions

    def forward(self, image):
        # Get visual features from the encoder
        image_graph, patch_predictions = self.image_to_graph(image)
        model_output = self.gnn_bottleneck(image_graph.x, image_graph.edge_index, image_graph.batch)
        if patch_predictions is not None:
            model_output["patch_predictions"] = patch_predictions
        if "node_embeddings" in model_output:
            node_embeddings = model_output["node_embeddings"].reshape(self.B, self.H*self.W, self.C)
            model_output["node_embeddings"] = node_embeddings
            node_embeddings_graph_processing = model_output["node_embeddings_graph_processing"].reshape(self.B, -1, self.C)
            model_output["node_embeddings_graph_processing"] = node_embeddings_graph_processing
        return model_output

    def print(self):
        patch_prediction_str=""
        if self.use_patch_predictions:
            patch_prediction_str = "_patch_predictions"
        return "i-WiViG_encoder={}_dilation={}_{}".format(self.visual_encoder.print(), self.dilation, self.gsat.print())