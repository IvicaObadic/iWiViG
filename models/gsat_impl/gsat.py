import math
import sys
sys.path.append('../src')

import scipy
import torch
import torch.nn as nn
#from torch_sparse import transpose
from torch_geometric.utils import is_undirected
from .utils.utils import reorder_like
from .utils.get_model import MLP
from torchvision.datasets import ImageNet
from torch_sparse import transpose
from torch.distributions import Categorical
from .utils.get_model import get_model

def calc_normalized_euclidean_distance_row_wise(f1, f2):
    diff = f1 - f2
    sq_diff = diff.pow(2)
    sum_sq_diff = sq_diff.sum(dim=1)
    distance = torch.sqrt(sum_sq_diff)

    min_val = distance.min()
    max_val = distance.max()

    # Avoid division by zero if all values are the same
    if max_val == min_val:
        return torch.zeros_like(distance)
    
    normalized_tensor = (distance - min_val) / (max_val - min_val)
    return normalized_tensor

class GSAT(nn.Module):
    def __init__(self, clf, extractor, learn_edge_att=True, final_r=0.7, decay_interval=10, decay_r=0.1, lambda_gsat_loss=1.0):
        super().__init__()
        self.clf = clf
        self.extractor = extractor

        self.learn_edge_att = learn_edge_att
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r
        self.add_edges = False
        self.descending = True
        self.insertion=True
        self.perc_to_add = .5
        self.explanation_edge_att = None
        self.lambda_gsat_loss = lambda_gsat_loss

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()
        
        loss = pred_loss + 0.1*info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def forward(self, data):
        edge_att = None
        att_log_logits = None

        if self.learn_edge_att:
            emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
            att_log_logits = self.extractor(emb, data.edge_index, data.batch)
            att = self.sampling(att_log_logits, self.training)

            if is_undirected(data.edge_index):
                print("Undirected graph: averaging edge attention scores", flush=True)
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                print("Directed graph: using edge attention scores as is", flush=True)
                edge_att = att

        if self.add_edges:
            if self.explanation_edge_att is None:
                edge_att_explanation = edge_att.squeeze()
            else:
                edge_att_explanation = self.explanation_edge_att
                edge_att = torch.ones_like(edge_att_explanation)
            
            idx = torch.argsort(edge_att_explanation, descending=self.descending)
            idx_to_take = idx[:math.ceil(self.perc_to_add * edge_att_explanation.shape[0])]
            print("Perc: {}, Number of edges to keep: {}".format(self.perc_to_add, len(idx_to_take)), flush=True)

            if self.insertion:
                mask = torch.zeros_like(edge_att_explanation)
                # Set the indices in idx_to_take to 1
                mask[idx_to_take] = 1.0
            else:
                mask = torch.ones_like(edge_att_explanation)
                # Set the indices in idx_to_take to 0
                mask[idx_to_take] = 0.0

            # Apply the mask to edge_att
            edge_att = edge_att * mask
            edge_att = torch.unsqueeze(edge_att, dim=1)

        #print(data.edge_index.shape, edge_att.shape)
        node_embeddings = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        return att_log_logits, edge_att, node_embeddings

    def print(self):
        info_loss_str = "no_info_loss"
        if self.lambda_gsat_loss > 0:
            info_loss_str = "with_info_loss"
        if not self.learn_edge_att:
            info_loss_str += "_no_learn_edge_att"

        return "GSAT_{}_{}".format(info_loss_str, self.clf.print())

    def sampling(self, att_log_logit, training):
        temp = 1
        if training and self.lambda_gsat_loss > 0:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        # att_bern = (att_log_logit).sigmoid()
        return att_bern

    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

def init_gsat_model(hidden_channels, num_layers, num_classes, gsat_r, lambda_gsat_loss=1.0, dropout_p=0.3, learn_edge_att=True):
    model_config = {'model_name': 'GIN', 'hidden_size': hidden_channels, 'n_layers': num_layers,
                     'dropout_p': dropout_p, 'use_edge_attr': False, "use_residuals": False, "residuals_mode": None, "pooling": False}
    gnn_backbone = get_model(hidden_channels, 0, num_classes, False, model_config)
    edge_extractor = ExtractorMLP(hidden_channels, learn_edge_att=True)
    gnn_model = GSAT(gnn_backbone, edge_extractor, learn_edge_att=learn_edge_att, final_r=gsat_r, lambda_gsat_loss=lambda_gsat_loss)
    return gnn_model

def gsat_loss(gsat_model, prediction_loss_fn, att, clf_logits, clf_labels, epoch, batch_size, lambda_info):
    pred_loss = prediction_loss_fn(clf_logits, clf_labels)
    gsat_graph = gsat_model.gnn_bottleneck.gnn_model

    r = gsat_graph.get_r(gsat_graph.decay_interval, gsat_graph.decay_r, epoch, final_r=gsat_graph.final_r)
    print("r: {}".format(r))
    info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()
    
    loss = pred_loss + (lambda_info*info_loss)
    loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
    return loss, loss_dict

class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, learn_edge_att):
        super().__init__()
        self.learn_edge_att = learn_edge_att

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=0.5)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=0.5)

    def forward(self, emb, edge_index, batch):
        node_distances = None
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits
